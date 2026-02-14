#!/usr/bin/env python3
"""
SA-1B stage-1 pipeline:
- CleanVision quality checks (blur, low-information, lighting, low-resolution)
- sha256 + phash
- parquet output

No YOLO inference in this script.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import multiprocessing as mp
import time
from pathlib import Path
from typing import Any, Dict, List

from utils.pipeline_utils import (
    build_cv_thresholds,
    list_tar_files,
    log,
    merge_parquet_parts,
    split_round_robin,
    worker_entry,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SA-1B stage-1: CleanVision + hash to parquet")
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
        default=Path("./sa1b_cv_hash_work"),
        help="Stores worker parquet shards before merge.",
    )
    parser.add_argument("--max-tars", type=int, default=0)
    parser.add_argument("--decode-batch-size", type=int, default=512)
    parser.add_argument(
        "--prefetch-tar-to-memory",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Read each tar file fully into RAM before extraction/decode.",
    )
    parser.add_argument(
        "--processes",
        type=int,
        default=2,
        help="Process count. Default is 2 as requested.",
    )
    parser.add_argument(
        "--cpu-threads-total",
        type=int,
        default=200,
        help="Total decode/hash threads across all processes.",
    )
    parser.add_argument("--cv-chunk-size", type=int, default=128)
    parser.add_argument("--min-width", type=int, default=256)
    parser.add_argument("--min-height", type=int, default=256)
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
    if args.processes <= 0:
        raise ValueError("--processes must be > 0")
    if args.cpu_threads_total <= 0:
        raise ValueError("--cpu-threads-total must be > 0")

    tar_files = list_tar_files(args.dataset_root, recursive=args.recursive)
    if args.max_tars > 0:
        tar_files = tar_files[: args.max_tars]
    if not tar_files:
        raise RuntimeError(f"No tar files found under {args.dataset_root}")

    worker_count = min(args.processes, len(tar_files))
    if worker_count <= 0:
        raise RuntimeError("No active workers available.")
    cpu_threads_per_worker = max(1, args.cpu_threads_total // worker_count)
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

    tar_buckets = split_round_robin(tar_files, worker_count)
    worker_payloads = []
    for worker_id, tar_bucket in enumerate(tar_buckets):
        if not tar_bucket:
            continue
        worker_payloads.append(
            (
                worker_id,
                worker_id,
                [str(p) for p in tar_bucket],
                {
                    "part_dir": str(part_dir),
                    "skip_yolo": True,
                    "yolo_model": "",
                    "head_class_ids": [],
                    "yolo_batch_size": 1,
                    "yolo_conf": 0.25,
                    "yolo_iou": 0.7,
                    "yolo_imgsz": 640,
                    "yolo_max_det": 1,
                    "yolo_half": False,
                    "decode_batch_size": args.decode_batch_size,
                    "prefetch_tar_to_memory": bool(args.prefetch_tar_to_memory),
                    "cpu_threads_per_worker": cpu_threads_per_worker,
                    "cv_workers": 1,
                    "cv_chunk_size": max(1, args.cv_chunk_size),
                    "min_width": args.min_width,
                    "min_height": args.min_height,
                    "blur_expand_ratio": 1.3,
                    "blur_ratio_threshold": 0.35,
                    "min_context_sharpness": 20.0,
                    "writer_mode": args.writer_mode,
                    "flush_rows": args.flush_rows,
                    "max_rows_in_memory": args.max_rows_in_memory,
                    "parquet_compression": args.parquet_compression,
                    "cv_thresholds": cv_thresholds,
                },
            )
        )

    if not worker_payloads:
        raise RuntimeError("No tar files assigned to workers.")

    log(
        f"Stage-1 start: tars={len(tar_files)} workers={len(worker_payloads)} "
        f"threads_total={args.cpu_threads_total} threads_per_worker={cpu_threads_per_worker} "
        f"prefetch_tar_to_memory={bool(args.prefetch_tar_to_memory)}"
    )
    log(
        f"Filters enabled: blurry, low_information, lighting(dark/light), "
        f"low_resolution(<{args.min_width}x{args.min_height})"
    )

    start = time.time()
    worker_results: List[Dict[str, Any]] = []
    ctx = mp.get_context("spawn")
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
        }
        for future in concurrent.futures.as_completed(future_map):
            worker_id, process_id = future_map[future]
            result = future.result()
            worker_results.append(result)
            log(
                f"Worker completed: id={worker_id} proc={process_id} "
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
        f"Stage-1 done. output={args.output_parquet} images={total_images} "
        f"decode_errors={total_decode_errors} elapsed={elapsed / 3600:.2f}h"
    )


if __name__ == "__main__":
    main()
