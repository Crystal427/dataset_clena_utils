#!/usr/bin/env python3
"""
SA-1B tar pipeline entrypoint.

Reusable logic lives in ./utils/pipeline_utils.py
"""

from __future__ import annotations

import argparse
import concurrent.futures
import multiprocessing as mp
import time
from pathlib import Path
from typing import Any, Dict, List

from utils.pipeline_utils import (
    DEFAULT_LOCAL_HEAD_MODEL,
    build_cv_thresholds,
    ensure_local_yolo_model,
    list_tar_files,
    log,
    merge_parquet_parts,
    parse_csv_ints,
    split_round_robin,
    validate_head_model,
    worker_entry,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SA-1B tar cleaning pipeline")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument(
        "--recursive",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--output-parquet", type=Path, required=True)
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=Path("./sa1b_pipeline_work"),
        help="Stores worker parquet shards before merge.",
    )
    parser.add_argument("--max-tars", type=int, default=0)
    parser.add_argument("--decode-batch-size", type=int, default=512)
    parser.add_argument("--cpu-threads", type=int, default=220)
    parser.add_argument(
        "--cv-workers",
        type=int,
        default=0,
        help=(
            "CleanVision multiprocessing workers per GPU worker. "
            "0 means auto: min(16, max(1, cpu_threads_per_worker//2))."
        ),
    )
    parser.add_argument(
        "--cv-chunk-size",
        type=int,
        default=128,
        help="Images per CleanVision multiprocessing task.",
    )
    parser.add_argument("--gpu-devices", type=str, default="4,5,6,7")
    parser.add_argument("--skip-yolo", action="store_true")
    parser.add_argument(
        "--yolo-model",
        type=str,
        default=DEFAULT_LOCAL_HEAD_MODEL,
        help="Local head model path. Default: <repo>/models/medium.pt.",
    )
    parser.add_argument(
        "--head-class-ids",
        type=str,
        default="0",
        help="Comma-separated class ids treated as head detections.",
    )
    parser.add_argument("--yolo-batch-size", type=int, default=256)
    parser.add_argument("--yolo-conf", type=float, default=0.25)
    parser.add_argument("--yolo-iou", type=float, default=0.7)
    parser.add_argument("--yolo-imgsz", type=int, default=640)
    parser.add_argument("--yolo-max-det", type=int, default=100)
    parser.add_argument(
        "--yolo-half",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--min-width", type=int, default=256)
    parser.add_argument("--min-height", type=int, default=256)
    parser.add_argument("--blur-expand-ratio", type=float, default=1.3)
    parser.add_argument("--blur-ratio-threshold", type=float, default=0.35)
    parser.add_argument("--min-context-sharpness", type=float, default=20.0)
    parser.add_argument(
        "--writer-mode",
        type=str,
        choices=["final", "buffered"],
        default="buffered",
        help="Use final only if RAM is definitely sufficient.",
    )
    parser.add_argument("--flush-rows", type=int, default=200_000)
    parser.add_argument(
        "--max-rows-in-memory",
        type=int,
        default=2_000_000,
        help="For writer_mode=final, auto-flush if this limit is reached. 0 disables.",
    )
    parser.add_argument("--parquet-compression", type=str, default="zstd")
    parser.add_argument("--dark-threshold", type=float, default=-1.0)
    parser.add_argument("--light-threshold", type=float, default=-1.0)
    parser.add_argument("--low-information-threshold", type=float, default=-1.0)
    parser.add_argument("--low-information-normalizing-factor", type=float, default=-1.0)
    parser.add_argument("--blurry-threshold", type=float, default=-1.0)
    parser.add_argument("--blurry-normalizing-factor", type=float, default=-1.0)
    parser.add_argument("--blurry-color-threshold", type=float, default=-1.0)
    parser.add_argument("--odd-size-threshold", type=float, default=-1.0)
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    if not args.dataset_root.exists():
        raise FileNotFoundError(f"dataset-root not found: {args.dataset_root}")

    tar_files = list_tar_files(args.dataset_root, recursive=args.recursive)
    if args.max_tars > 0:
        tar_files = tar_files[: args.max_tars]
    if not tar_files:
        raise RuntimeError(f"No tar files found under {args.dataset_root}")

    gpu_ids = parse_csv_ints(args.gpu_devices)
    if not gpu_ids:
        raise ValueError("gpu-devices cannot be empty")

    if args.skip_yolo and len(gpu_ids) > 1:
        log("skip-yolo enabled, but multiple GPU workers requested; keeping workers as-is.")

    validate_head_model(args.yolo_model, args.skip_yolo)

    resolved_yolo_model = args.yolo_model
    if not args.skip_yolo:
        resolved_yolo_model = ensure_local_yolo_model(args.yolo_model)

    head_class_ids = parse_csv_ints(args.head_class_ids) if args.head_class_ids else []
    cpu_threads_per_worker = max(1, args.cpu_threads // len(gpu_ids))
    if args.cv_workers <= 0:
        cv_workers = min(16, max(1, cpu_threads_per_worker // 2))
    else:
        cv_workers = max(1, args.cv_workers)
    cv_chunk_size = max(1, args.cv_chunk_size)

    cv_thresholds = build_cv_thresholds(
        dark_threshold=args.dark_threshold,
        light_threshold=args.light_threshold,
        low_information_threshold=args.low_information_threshold,
        low_information_normalizing_factor=args.low_information_normalizing_factor,
        blurry_threshold=args.blurry_threshold,
        blurry_normalizing_factor=args.blurry_normalizing_factor,
        blurry_color_threshold=args.blurry_color_threshold,
        odd_size_threshold=args.odd_size_threshold,
    )

    args.work_dir.mkdir(parents=True, exist_ok=True)
    part_dir = args.work_dir / "parquet_parts"
    part_dir.mkdir(parents=True, exist_ok=True)

    tar_buckets = split_round_robin(tar_files, len(gpu_ids))
    worker_payloads = []
    for worker_id, (gpu_id, tar_bucket) in enumerate(zip(gpu_ids, tar_buckets)):
        worker_payloads.append(
            (
                worker_id,
                gpu_id,
                [str(p) for p in tar_bucket],
                {
                    "part_dir": str(part_dir),
                    "skip_yolo": bool(args.skip_yolo),
                    "yolo_model": resolved_yolo_model,
                    "head_class_ids": head_class_ids,
                    "yolo_batch_size": args.yolo_batch_size,
                    "yolo_conf": args.yolo_conf,
                    "yolo_iou": args.yolo_iou,
                    "yolo_imgsz": args.yolo_imgsz,
                    "yolo_max_det": args.yolo_max_det,
                    "yolo_half": bool(args.yolo_half),
                    "decode_batch_size": args.decode_batch_size,
                    "cpu_threads_per_worker": cpu_threads_per_worker,
                    "cv_workers": cv_workers,
                    "cv_chunk_size": cv_chunk_size,
                    "min_width": args.min_width,
                    "min_height": args.min_height,
                    "blur_expand_ratio": args.blur_expand_ratio,
                    "blur_ratio_threshold": args.blur_ratio_threshold,
                    "min_context_sharpness": args.min_context_sharpness,
                    "writer_mode": args.writer_mode,
                    "flush_rows": args.flush_rows,
                    "max_rows_in_memory": args.max_rows_in_memory,
                    "parquet_compression": args.parquet_compression,
                    "cv_thresholds": cv_thresholds,
                },
            )
        )

    log(
        f"Found {len(tar_files)} tar files, workers={len(worker_payloads)}, "
        f"cpu_threads_total={args.cpu_threads}, cpu_threads_per_worker={cpu_threads_per_worker}, "
        f"cv_workers_per_worker={cv_workers}, cv_chunk_size={cv_chunk_size}"
    )
    log(
        f"YOLO model={resolved_yolo_model}, gpu_devices={gpu_ids}, "
        f"head_class_ids={head_class_ids if head_class_ids else 'ALL'}"
    )
    log(
        f"CleanVision thresholds: dark={cv_thresholds['dark_threshold']} "
        f"light={cv_thresholds['light_threshold']} "
        f"low_information={cv_thresholds['low_information_threshold']} "
        f"blurry={cv_thresholds['blurry_threshold']} odd_size={cv_thresholds['odd_size_threshold']}"
    )

    start = time.time()
    ctx = mp.get_context("spawn")
    worker_results: List[Dict[str, Any]] = []
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=len(worker_payloads),
        mp_context=ctx,
    ) as pool:
        future_map = {
            pool.submit(worker_entry, payload[0], payload[1], payload[2], payload[3]): (
                payload[0],
                payload[1],
            )
            for payload in worker_payloads
            if payload[2]
        }

        if not future_map:
            raise RuntimeError("No tar files assigned to workers.")

        for future in concurrent.futures.as_completed(future_map):
            worker_id, gpu_id = future_map[future]
            result = future.result()
            worker_results.append(result)
            log(
                f"Worker completed: id={worker_id} gpu={gpu_id} "
                f"images={result['image_count']} decode_errors={result['decode_error_count']} "
                f"elapsed={result['elapsed_sec']:.1f}s"
            )

    part_paths = sorted(Path(r["part_path"]) for r in worker_results if r["image_count"] > 0)
    if not part_paths:
        raise RuntimeError("No output rows were generated.")

    merge_parquet_parts(
        part_paths=part_paths,
        output_parquet=args.output_parquet,
        compression=args.parquet_compression,
    )

    elapsed = time.time() - start
    total_images = sum(r["image_count"] for r in worker_results)
    total_decode_errors = sum(r["decode_error_count"] for r in worker_results)
    log(
        f"Done. output={args.output_parquet} images={total_images} "
        f"decode_errors={total_decode_errors} elapsed={elapsed / 3600:.2f}h"
    )


if __name__ == "__main__":
    main()
