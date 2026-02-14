#!/usr/bin/env python3
"""
SA-1B test pipeline (100k-scale) with optional Hugging Face upload.

This script is independent from the full distributed pipeline and is intended
for quick validation runs.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from ultralytics import YOLO

from utils.pipeline_utils import (
    CVThresholds,
    CleanVisionBatchScorer,
    DEFAULT_LOCAL_HEAD_MODEL,
    ParquetRowSink,
    build_cv_thresholds,
    ensure_local_yolo_model,
    iter_tar_images,
    list_tar_files,
    log,
    parse_csv_ints,
    process_batch,
    resolve_torch_cuda_index,
    validate_head_model,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SA-1B 100k test pipeline + HF upload")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument(
        "--recursive",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--output-parquet", type=Path, required=True)
    parser.add_argument("--max-images", type=int, default=100_000)
    parser.add_argument("--cpu-threads", type=int, default=64)
    parser.add_argument("--gpu-device", type=int, default=4)
    parser.add_argument("--decode-batch-size", type=int, default=512)
    parser.add_argument("--skip-yolo", action="store_true")
    parser.add_argument(
        "--yolo-model",
        type=str,
        default=DEFAULT_LOCAL_HEAD_MODEL,
        help="Local head model path. Default: <repo>/models/medium.pt.",
    )
    parser.add_argument("--head-class-ids", type=str, default="0")
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
    parser.add_argument("--flush-rows", type=int, default=50_000)
    parser.add_argument("--parquet-compression", type=str, default="zstd")
    parser.add_argument("--dark-threshold", type=float, default=-1.0)
    parser.add_argument("--light-threshold", type=float, default=-1.0)
    parser.add_argument("--low-information-threshold", type=float, default=-1.0)
    parser.add_argument("--low-information-normalizing-factor", type=float, default=-1.0)
    parser.add_argument("--blurry-threshold", type=float, default=-1.0)
    parser.add_argument("--blurry-normalizing-factor", type=float, default=-1.0)
    parser.add_argument("--blurry-color-threshold", type=float, default=-1.0)
    parser.add_argument("--odd-size-threshold", type=float, default=-1.0)
    parser.add_argument(
        "--upload-to-hf",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--hf-repo-id", type=str, default="")
    parser.add_argument(
        "--hf-token",
        type=str,
        default="",
        help="Required when upload is enabled. Must be explicitly provided.",
    )
    parser.add_argument(
        "--hf-private",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--hf-path-prefix", type=str, default="sa1b_test")
    return parser


def iter_tar_items(tar_files: List[Path]) -> Iterable[Tuple[str, str, bytes]]:
    for tar_path in tar_files:
        for member_name, payload in iter_tar_images(tar_path):
            yield str(tar_path), member_name, payload


def upload_to_hf(
    parquet_path: Path,
    stats_path: Path,
    repo_id: str,
    token: str,
    private: bool,
    path_prefix: str,
) -> None:
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:  # noqa: BLE001
        raise ImportError(
            "huggingface_hub is required for upload. "
            "Install it via `pip install huggingface_hub`."
        ) from exc

    if not token.strip():
        raise ValueError(
            "Hugging Face upload requires explicit --hf-token. "
            "Please pass --hf-token hf_xxx."
        )
    if not repo_id.strip():
        raise ValueError("Hugging Face upload requires --hf-repo-id (e.g. user/repo).")

    api = HfApi(token=token)
    api.create_repo(
        repo_id=repo_id,
        repo_type="dataset",
        private=private,
        exist_ok=True,
        token=token,
    )

    parquet_remote = f"{path_prefix.strip('/')}/{parquet_path.name}"
    stats_remote = f"{path_prefix.strip('/')}/{stats_path.name}"
    api.upload_file(
        path_or_fileobj=str(parquet_path),
        path_in_repo=parquet_remote,
        repo_id=repo_id,
        repo_type="dataset",
        token=token,
        commit_message=f"Upload SA-1B test parquet ({parquet_path.name})",
    )
    api.upload_file(
        path_or_fileobj=str(stats_path),
        path_in_repo=stats_remote,
        repo_id=repo_id,
        repo_type="dataset",
        token=token,
        commit_message=f"Upload SA-1B test stats ({stats_path.name})",
    )
    log(f"Uploaded to HF dataset repo: {repo_id} -> {parquet_remote}")


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.max_images <= 0:
        raise ValueError("--max-images must be > 0")
    if not args.dataset_root.exists():
        raise FileNotFoundError(f"dataset-root not found: {args.dataset_root}")

    if args.upload_to_hf and not args.hf_token.strip():
        raise ValueError(
            "Upload is enabled, but --hf-token is missing. "
            "For safety, token must be explicitly provided."
        )

    tar_files = list_tar_files(args.dataset_root, recursive=args.recursive)
    if not tar_files:
        raise RuntimeError(f"No tar files found under {args.dataset_root}")

    validate_head_model(args.yolo_model, args.skip_yolo)
    resolved_yolo_model = args.yolo_model
    if not args.skip_yolo:
        resolved_yolo_model = ensure_local_yolo_model(args.yolo_model)

    head_class_ids = parse_csv_ints(args.head_class_ids) if args.head_class_ids else []
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

    local_cuda_index = resolve_torch_cuda_index(args.gpu_device)
    yolo_model = None if args.skip_yolo else YOLO(resolved_yolo_model)
    yolo_device = f"cuda:{local_cuda_index}"
    log(f"GPU mapping: requested={args.gpu_device} -> torch_device={yolo_device}")
    scorer = CleanVisionBatchScorer(CVThresholds(**cv_thresholds))
    args.output_parquet.parent.mkdir(parents=True, exist_ok=True)
    sink = ParquetRowSink(
        output_path=args.output_parquet,
        compression=args.parquet_compression,
        writer_mode="buffered",
        flush_rows=args.flush_rows,
        max_rows_in_memory=0,
    )

    stats = {
        "processed_rows": 0,
        "decode_error_rows": 0,
        "quality_issue_rows": 0,
        "head_blur_anomaly_rows": 0,
        "max_images": int(args.max_images),
        "yolo_model": resolved_yolo_model,
        "dataset_root": str(args.dataset_root),
        "output_parquet": str(args.output_parquet),
    }

    start = time.time()
    batch: List[Tuple[str, str, bytes]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.cpu_threads) as decode_pool:
        for item in iter_tar_items(tar_files):
            batch.append(item)
            if len(batch) < args.decode_batch_size:
                continue

            rows = process_batch(
                worker_id=0,
                gpu_id=args.gpu_device,
                decode_pool=decode_pool,
                cleanvision_scorer=scorer,
                yolo_model=yolo_model,
                yolo_device=yolo_device,
                batch=batch,
                min_width=args.min_width,
                min_height=args.min_height,
                yolo_batch_size=args.yolo_batch_size,
                yolo_conf=args.yolo_conf,
                yolo_iou=args.yolo_iou,
                yolo_imgsz=args.yolo_imgsz,
                yolo_max_det=args.yolo_max_det,
                yolo_half=args.yolo_half,
                head_class_ids=head_class_ids,
                blur_expand_ratio=args.blur_expand_ratio,
                blur_ratio_threshold=args.blur_ratio_threshold,
                min_context_sharpness=args.min_context_sharpness,
            )
            remaining = args.max_images - stats["processed_rows"]
            rows = rows[:remaining]
            sink.add_rows(rows)
            stats["processed_rows"] += len(rows)
            stats["decode_error_rows"] += sum(1 for row in rows if row["decode_error"])
            stats["quality_issue_rows"] += sum(1 for row in rows if row["is_quality_issue"])
            stats["head_blur_anomaly_rows"] += sum(
                1 for row in rows if row["is_any_head_blur_anomaly"]
            )
            batch.clear()

            if stats["processed_rows"] >= args.max_images:
                break

        if batch and stats["processed_rows"] < args.max_images:
            rows = process_batch(
                worker_id=0,
                gpu_id=args.gpu_device,
                decode_pool=decode_pool,
                cleanvision_scorer=scorer,
                yolo_model=yolo_model,
                yolo_device=yolo_device,
                batch=batch,
                min_width=args.min_width,
                min_height=args.min_height,
                yolo_batch_size=args.yolo_batch_size,
                yolo_conf=args.yolo_conf,
                yolo_iou=args.yolo_iou,
                yolo_imgsz=args.yolo_imgsz,
                yolo_max_det=args.yolo_max_det,
                yolo_half=args.yolo_half,
                head_class_ids=head_class_ids,
                blur_expand_ratio=args.blur_expand_ratio,
                blur_ratio_threshold=args.blur_ratio_threshold,
                min_context_sharpness=args.min_context_sharpness,
            )
            remaining = args.max_images - stats["processed_rows"]
            rows = rows[:remaining]
            sink.add_rows(rows)
            stats["processed_rows"] += len(rows)
            stats["decode_error_rows"] += sum(1 for row in rows if row["decode_error"])
            stats["quality_issue_rows"] += sum(1 for row in rows if row["is_quality_issue"])
            stats["head_blur_anomaly_rows"] += sum(
                1 for row in rows if row["is_any_head_blur_anomaly"]
            )

    sink.close()
    elapsed = time.time() - start
    stats["elapsed_sec"] = elapsed
    stats["images_per_sec"] = round(
        stats["processed_rows"] / elapsed, 3
    ) if elapsed > 0 else 0.0

    stats_path = args.output_parquet.with_suffix(".stats.json")
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    log(
        f"Test pipeline done: rows={stats['processed_rows']} "
        f"decode_errors={stats['decode_error_rows']} "
        f"elapsed={elapsed:.1f}s output={args.output_parquet}"
    )

    if args.upload_to_hf:
        upload_to_hf(
            parquet_path=args.output_parquet,
            stats_path=stats_path,
            repo_id=args.hf_repo_id,
            token=args.hf_token,
            private=args.hf_private,
            path_prefix=args.hf_path_prefix,
        )


if __name__ == "__main__":
    main()
