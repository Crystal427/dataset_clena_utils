#!/usr/bin/env python3
"""
SA-1B tar pipeline:
1) Stream images directly from tar shards
2) Compute SHA256 + pHash
3) Run CleanVision-style quality checks (blurry, low information, lighting, odd size)
4) Run YOLOv11n head/person detection on selected GPUs
5) Record bbox + head blur-context analysis
6) Write a Parquet dataset

Notes:
- For SA-1B scale, writing only once at the very end is typically not RAM-safe.
  Use buffered mode unless you are sure memory is sufficient.
- `yolo11n.pt` is a COCO detector (no explicit head class). If you need true head
  detection, pass a head-trained model via --yolo-model and set --head-class-ids.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import io
import json
import math
import multiprocessing as mp
import os
import tarfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import imagehash
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image, ImageFile
from ultralytics import YOLO

from cleanvision.issue_managers import IssueType
from cleanvision.issue_managers.image_property import (
    BlurrinessProperty,
    BrightnessProperty,
    EntropyProperty,
    SizeProperty,
)
from cleanvision.issue_managers.image_property_issue_manager import (
    ImagePropertyIssueManager,
)
from cleanvision.utils.utils import get_is_issue_colname, get_score_colname

ImageFile.LOAD_TRUNCATED_IMAGES = True

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp",
    ".tif",
    ".tiff",
}


def ts() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def log(msg: str) -> None:
    print(f"[{ts()}] {msg}", flush=True)


@dataclass
class CVThresholds:
    dark_threshold: float
    light_threshold: float
    low_information_threshold: float
    low_information_normalizing_factor: float
    blurry_threshold: float
    blurry_normalizing_factor: float
    blurry_color_threshold: float
    odd_size_threshold: float


class CleanVisionBatchScorer:
    def __init__(self, thresholds: CVThresholds):
        self.thresholds = thresholds
        self.dark_property = BrightnessProperty(IssueType.DARK.value)
        self.light_property = BrightnessProperty(IssueType.LIGHT.value)
        self.entropy_property = EntropyProperty()
        self.blur_property = BlurrinessProperty()
        self.size_property = SizeProperty()

    def score_batch(self, pil_images: List[Image.Image]) -> List[Dict[str, Any]]:
        if not pil_images:
            return []

        raw_rows: List[Dict[str, Any]] = []
        for image in pil_images:
            row: Dict[str, Any] = {}
            row.update(self.dark_property.calculate(image))
            row.update(self.entropy_property.calculate(image))
            row.update(self.blur_property.calculate(image))
            row.update(self.size_property.calculate(image))
            raw_rows.append(row)

        raw_df = pd.DataFrame(raw_rows)

        dark_scores = self.dark_property.get_scores(
            raw_df[self.dark_property.score_columns], issue_type=IssueType.DARK.value
        )
        dark_flags = self.dark_property.mark_issue(
            dark_scores,
            threshold=self.thresholds.dark_threshold,
            issue_type=IssueType.DARK.value,
        )

        light_scores = self.light_property.get_scores(
            raw_df[self.light_property.score_columns], issue_type=IssueType.LIGHT.value
        )
        light_flags = self.light_property.mark_issue(
            light_scores,
            threshold=self.thresholds.light_threshold,
            issue_type=IssueType.LIGHT.value,
        )

        low_info_scores = self.entropy_property.get_scores(
            raw_df[self.entropy_property.score_columns],
            issue_type=IssueType.LOW_INFORMATION.value,
            normalizing_factor=self.thresholds.low_information_normalizing_factor,
        )
        low_info_flags = self.entropy_property.mark_issue(
            low_info_scores,
            threshold=self.thresholds.low_information_threshold,
            issue_type=IssueType.LOW_INFORMATION.value,
        )

        blurry_scores = self.blur_property.get_scores(
            raw_df[self.blur_property.score_columns],
            issue_type=IssueType.BLURRY.value,
            normalizing_factor=self.thresholds.blurry_normalizing_factor,
            color_threshold=self.thresholds.blurry_color_threshold,
        )
        blurry_flags = self.blur_property.mark_issue(
            blurry_scores,
            threshold=self.thresholds.blurry_threshold,
            issue_type=IssueType.BLURRY.value,
        )

        odd_size_scores = self.size_property.get_scores(
            raw_df[self.size_property.score_columns], issue_type=IssueType.ODD_SIZE.value
        )
        odd_size_flags = self.size_property.mark_issue(
            odd_size_scores,
            threshold=self.thresholds.odd_size_threshold,
            issue_type=IssueType.ODD_SIZE.value,
        )

        dark_score_col = get_score_colname(IssueType.DARK.value)
        dark_issue_col = get_is_issue_colname(IssueType.DARK.value)
        light_score_col = get_score_colname(IssueType.LIGHT.value)
        light_issue_col = get_is_issue_colname(IssueType.LIGHT.value)
        low_info_score_col = get_score_colname(IssueType.LOW_INFORMATION.value)
        low_info_issue_col = get_is_issue_colname(IssueType.LOW_INFORMATION.value)
        blurry_score_col = get_score_colname(IssueType.BLURRY.value)
        blurry_issue_col = get_is_issue_colname(IssueType.BLURRY.value)
        odd_size_score_col = get_score_colname(IssueType.ODD_SIZE.value)
        odd_size_issue_col = get_is_issue_colname(IssueType.ODD_SIZE.value)

        out: List[Dict[str, Any]] = []
        for idx in range(len(pil_images)):
            is_dark = bool(dark_flags.at[idx, dark_issue_col])
            is_light = bool(light_flags.at[idx, light_issue_col])
            item = {
                dark_score_col: float(dark_scores.at[idx, dark_score_col]),
                dark_issue_col: is_dark,
                light_score_col: float(light_scores.at[idx, light_score_col]),
                light_issue_col: is_light,
                low_info_score_col: float(low_info_scores.at[idx, low_info_score_col]),
                low_info_issue_col: bool(low_info_flags.at[idx, low_info_issue_col]),
                blurry_score_col: float(blurry_scores.at[idx, blurry_score_col]),
                blurry_issue_col: bool(blurry_flags.at[idx, blurry_issue_col]),
                odd_size_score_col: float(odd_size_scores.at[idx, odd_size_score_col]),
                odd_size_issue_col: bool(odd_size_flags.at[idx, odd_size_issue_col]),
                "is_lighting_issue": is_dark or is_light,
            }
            out.append(item)
        return out


class ParquetRowSink:
    def __init__(
        self,
        output_path: Path,
        compression: str,
        writer_mode: str,
        flush_rows: int,
        max_rows_in_memory: int,
    ):
        self.output_path = output_path
        self.compression = compression
        self.writer_mode = writer_mode
        self.flush_rows = flush_rows
        self.max_rows_in_memory = max_rows_in_memory
        self._rows: List[Dict[str, Any]] = []
        self._writer: Optional[pq.ParquetWriter] = None
        self.row_count = 0
        self.flush_count = 0

    def add_rows(self, rows: List[Dict[str, Any]]) -> None:
        if not rows:
            return
        self._rows.extend(rows)

        should_flush = False
        if self.writer_mode == "buffered" and len(self._rows) >= self.flush_rows:
            should_flush = True
        elif self.writer_mode == "final" and self.max_rows_in_memory > 0:
            if len(self._rows) >= self.max_rows_in_memory:
                should_flush = True

        if should_flush:
            self.flush()

    def flush(self) -> None:
        if not self._rows:
            return
        table = pa.Table.from_pylist(self._rows)
        if self._writer is None:
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            self._writer = pq.ParquetWriter(
                where=self.output_path, schema=table.schema, compression=self.compression
            )
        self._writer.write_table(table)
        self.row_count += len(self._rows)
        self._rows.clear()
        self.flush_count += 1

    def close(self) -> None:
        self.flush()
        if self._writer is not None:
            self._writer.close()
            self._writer = None


def parse_csv_ints(csv_text: str) -> List[int]:
    return [int(x.strip()) for x in csv_text.split(",") if x.strip()]


def list_tar_files(root: Path, recursive: bool) -> List[Path]:
    pattern = "**/*.tar" if recursive else "*.tar"
    return sorted(root.glob(pattern))


def split_round_robin(items: Sequence[Path], bucket_count: int) -> List[List[Path]]:
    buckets: List[List[Path]] = [[] for _ in range(bucket_count)]
    for i, item in enumerate(items):
        buckets[i % bucket_count].append(item)
    return buckets


def decode_hash_image(
    tar_path: str, member_name: str, payload: bytes
) -> Dict[str, Any]:
    sha256_hex = hashlib.sha256(payload).hexdigest()
    out: Dict[str, Any] = {
        "ok": False,
        "tar_path": tar_path,
        "member_name": member_name,
        "sha256": sha256_hex,
        "phash": None,
        "width": None,
        "height": None,
        "pil_rgb": None,
        "image_bgr": None,
        "error_message": None,
    }
    try:
        with Image.open(io.BytesIO(payload)) as img:
            rgb = img.convert("RGB")
            width, height = rgb.size
            phash_hex = str(imagehash.phash(rgb))
            rgb_np = np.asarray(rgb)

        out.update(
            {
                "ok": True,
                "phash": phash_hex,
                "width": int(width),
                "height": int(height),
                "pil_rgb": rgb,
                "image_bgr": cv2.cvtColor(rgb_np, cv2.COLOR_RGB2BGR),
            }
        )
    except Exception as exc:  # noqa: BLE001
        out["error_message"] = f"{type(exc).__name__}: {exc}"

    return out


def _parse_yolo_boxes(
    result: Any, head_class_ids: Optional[set[int]]
) -> List[Dict[str, Any]]:
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return []

    xyxy = boxes.xyxy.detach().cpu().numpy()
    confs = boxes.conf.detach().cpu().numpy()
    classes = boxes.cls.detach().cpu().numpy().astype(np.int32)
    detections: List[Dict[str, Any]] = []
    for i in range(len(xyxy)):
        cls_id = int(classes[i])
        if head_class_ids and cls_id not in head_class_ids:
            continue
        x1, y1, x2, y2 = xyxy[i].tolist()
        detections.append(
            {
                "class_id": cls_id,
                "conf": float(confs[i]),
                "bbox_xyxy": [float(x1), float(y1), float(x2), float(y2)],
            }
        )
    return detections


def analyze_head_blur(
    image_bgr: np.ndarray,
    detections: List[Dict[str, Any]],
    expand_ratio: float,
    ratio_threshold: float,
    min_context_sharpness: float,
) -> Tuple[List[Dict[str, Any]], int]:
    if not detections:
        return [], 0

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    analyses: List[Dict[str, Any]] = []
    anomaly_count = 0

    for det in detections:
        x1f, y1f, x2f, y2f = det["bbox_xyxy"]
        x1 = max(0, min(w - 1, int(math.floor(x1f))))
        y1 = max(0, min(h - 1, int(math.floor(y1f))))
        x2 = max(0, min(w, int(math.ceil(x2f))))
        y2 = max(0, min(h, int(math.ceil(y2f))))
        if x2 <= x1 or y2 <= y1:
            continue

        bw, bh = x2 - x1, y2 - y1
        if bw < 8 or bh < 8:
            analyses.append(
                {
                    "class_id": det["class_id"],
                    "conf": det["conf"],
                    "bbox_xyxy": det["bbox_xyxy"],
                    "face_sharpness": None,
                    "context_sharpness": None,
                    "sharpness_ratio": None,
                    "is_head_blur_anomaly": False,
                }
            )
            continue

        exp_w = int(round(bw * (expand_ratio - 1.0) * 0.5))
        exp_h = int(round(bh * (expand_ratio - 1.0) * 0.5))
        ex1 = max(0, x1 - exp_w)
        ey1 = max(0, y1 - exp_h)
        ex2 = min(w, x2 + exp_w)
        ey2 = min(h, y2 + exp_h)

        outer = gray[ey1:ey2, ex1:ex2]
        if outer.size == 0:
            continue

        lx1, ly1 = x1 - ex1, y1 - ey1
        lx2, ly2 = x2 - ex1, y2 - ey1

        lap = cv2.Laplacian(outer, cv2.CV_64F)
        face_lap = lap[ly1:ly2, lx1:lx2]
        ring_mask = np.ones(lap.shape, dtype=bool)
        ring_mask[ly1:ly2, lx1:lx2] = False
        ring_vals = lap[ring_mask]

        face_sharpness = float(face_lap.var()) if face_lap.size else 0.0
        context_sharpness = float(ring_vals.var()) if ring_vals.size else 0.0
        sharpness_ratio = float(face_sharpness / (context_sharpness + 1e-6))
        is_anomaly = bool(
            sharpness_ratio < ratio_threshold
            and context_sharpness >= min_context_sharpness
        )
        anomaly_count += int(is_anomaly)

        analyses.append(
            {
                "class_id": det["class_id"],
                "conf": det["conf"],
                "bbox_xyxy": det["bbox_xyxy"],
                "face_sharpness": face_sharpness,
                "context_sharpness": context_sharpness,
                "sharpness_ratio": sharpness_ratio,
                "is_head_blur_anomaly": is_anomaly,
            }
        )

    return analyses, anomaly_count


def make_base_row(
    worker_id: int,
    gpu_id: int,
    tar_path: str,
    member_name: str,
) -> Dict[str, Any]:
    sample_id = f"{tar_path}::{member_name}"
    return {
        "sample_id": sample_id,
        "tar_path": tar_path,
        "member_name": member_name,
        "worker_id": worker_id,
        "gpu_id": gpu_id,
        "width": None,
        "height": None,
        "sha256": None,
        "phash": None,
        "dark_score": None,
        "is_dark_issue": False,
        "light_score": None,
        "is_light_issue": False,
        "low_information_score": None,
        "is_low_information_issue": False,
        "blurry_score": None,
        "is_blurry_issue": False,
        "odd_size_score": None,
        "is_odd_size_issue": False,
        "is_lighting_issue": False,
        "is_low_resolution_issue": False,
        "is_quality_issue": False,
        "head_count": 0,
        "max_head_conf": 0.0,
        "head_bboxes_json": "[]",
        "head_analysis_json": "[]",
        "head_blur_anomaly_count": 0,
        "is_any_head_blur_anomaly": False,
        "decode_error": False,
        "error_message": None,
        "processed_at": ts(),
    }


def _run_yolo_predict(
    model: YOLO,
    images_bgr: List[np.ndarray],
    device: str,
    yolo_batch_size: int,
    yolo_conf: float,
    yolo_iou: float,
    yolo_imgsz: int,
    yolo_max_det: int,
    yolo_half: bool,
    head_class_ids: Optional[List[int]],
) -> List[Any]:
    if not images_bgr:
        return []

    classes_arg = head_class_ids if head_class_ids else None
    results: List[Any] = []
    start = 0
    dynamic_batch = yolo_batch_size

    while start < len(images_bgr):
        end = min(start + dynamic_batch, len(images_bgr))
        chunk = images_bgr[start:end]
        try:
            pred = model.predict(
                source=chunk,
                device=device,
                conf=yolo_conf,
                iou=yolo_iou,
                imgsz=yolo_imgsz,
                max_det=yolo_max_det,
                half=yolo_half,
                classes=classes_arg,
                verbose=False,
            )
            results.extend(pred)
            start = end
        except RuntimeError as exc:  # noqa: BLE001
            if "out of memory" not in str(exc).lower() or dynamic_batch <= 1:
                raise
            dynamic_batch = max(1, dynamic_batch // 2)
            log(f"YOLO OOM on {device}, reducing batch size to {dynamic_batch}")

    return results


def process_batch(
    worker_id: int,
    gpu_id: int,
    decode_pool: concurrent.futures.ThreadPoolExecutor,
    cleanvision_scorer: CleanVisionBatchScorer,
    yolo_model: Optional[YOLO],
    yolo_device: str,
    batch: List[Tuple[str, str, bytes]],
    min_width: int,
    min_height: int,
    yolo_batch_size: int,
    yolo_conf: float,
    yolo_iou: float,
    yolo_imgsz: int,
    yolo_max_det: int,
    yolo_half: bool,
    head_class_ids: Optional[List[int]],
    blur_expand_ratio: float,
    blur_ratio_threshold: float,
    min_context_sharpness: float,
) -> List[Dict[str, Any]]:
    decoded = list(
        decode_pool.map(
            lambda item: decode_hash_image(item[0], item[1], item[2]),
            batch,
        )
    )

    rows: List[Dict[str, Any]] = []
    valid_items: List[Dict[str, Any]] = []
    valid_indices: List[int] = []

    for idx, dec in enumerate(decoded):
        base = make_base_row(worker_id, gpu_id, dec["tar_path"], dec["member_name"])
        base["sha256"] = dec["sha256"]
        base["phash"] = dec["phash"]
        base["width"] = dec["width"]
        base["height"] = dec["height"]
        if not dec["ok"]:
            base["decode_error"] = True
            base["error_message"] = dec["error_message"]
            rows.append(base)
            continue

        valid_items.append(dec)
        valid_indices.append(idx)
        rows.append(base)

    if not valid_items:
        return rows

    pil_images = [item["pil_rgb"] for item in valid_items]
    bgr_images = [item["image_bgr"] for item in valid_items]

    cv_metrics = cleanvision_scorer.score_batch(pil_images)

    yolo_results: List[Any]
    if yolo_model is None:
        yolo_results = [None] * len(valid_items)
    else:
        yolo_results = _run_yolo_predict(
            model=yolo_model,
            images_bgr=bgr_images,
            device=yolo_device,
            yolo_batch_size=yolo_batch_size,
            yolo_conf=yolo_conf,
            yolo_iou=yolo_iou,
            yolo_imgsz=yolo_imgsz,
            yolo_max_det=yolo_max_det,
            yolo_half=yolo_half,
            head_class_ids=head_class_ids,
        )

    class_filter_set = set(head_class_ids) if head_class_ids else None
    for local_idx, (dec, cvm) in enumerate(zip(valid_items, cv_metrics)):
        row = rows[valid_indices[local_idx]]
        row.update(cvm)

        width = int(dec["width"])
        height = int(dec["height"])
        row["is_low_resolution_issue"] = bool(width < min_width or height < min_height)
        row["is_quality_issue"] = bool(
            row["is_blurry_issue"]
            or row["is_low_information_issue"]
            or row["is_lighting_issue"]
            or row["is_odd_size_issue"]
            or row["is_low_resolution_issue"]
        )

        if yolo_model is None:
            continue

        detections = _parse_yolo_boxes(yolo_results[local_idx], class_filter_set)
        analyses, anomaly_count = analyze_head_blur(
            image_bgr=dec["image_bgr"],
            detections=detections,
            expand_ratio=blur_expand_ratio,
            ratio_threshold=blur_ratio_threshold,
            min_context_sharpness=min_context_sharpness,
        )
        row["head_count"] = int(len(detections))
        row["max_head_conf"] = (
            float(max(d["conf"] for d in detections)) if detections else 0.0
        )
        row["head_bboxes_json"] = json.dumps(detections, separators=(",", ":"))
        row["head_analysis_json"] = json.dumps(analyses, separators=(",", ":"))
        row["head_blur_anomaly_count"] = int(anomaly_count)
        row["is_any_head_blur_anomaly"] = bool(anomaly_count > 0)

    return rows


def iter_tar_images(tar_path: Path) -> Iterable[Tuple[str, bytes]]:
    with tarfile.open(tar_path, "r:*") as tar:
        for member in tar:
            if not member.isfile():
                continue
            suffix = Path(member.name).suffix.lower()
            if suffix not in IMAGE_EXTENSIONS:
                continue
            file_obj = tar.extractfile(member)
            if file_obj is None:
                continue
            payload = file_obj.read()
            if not payload:
                continue
            yield member.name, payload


def worker_entry(
    worker_id: int,
    gpu_id: int,
    tar_paths: List[str],
    args_dict: Dict[str, Any],
) -> Dict[str, Any]:
    worker_start = time.time()
    worker_name = f"W{worker_id}-GPU{gpu_id}"
    log(f"{worker_name} starting with {len(tar_paths)} tar files")

    yolo_device = f"cuda:{gpu_id}"
    yolo_model: Optional[YOLO] = None
    if not args_dict["skip_yolo"]:
        yolo_model = YOLO(args_dict["yolo_model"])

    thresholds = CVThresholds(**args_dict["cv_thresholds"])
    cleanvision_scorer = CleanVisionBatchScorer(thresholds)

    part_path = Path(args_dict["part_dir"]) / f"part-worker-{worker_id:02d}.parquet"
    sink = ParquetRowSink(
        output_path=part_path,
        compression=args_dict["parquet_compression"],
        writer_mode=args_dict["writer_mode"],
        flush_rows=args_dict["flush_rows"],
        max_rows_in_memory=args_dict["max_rows_in_memory"],
    )

    total_images = 0
    total_decode_errors = 0
    total_tars = 0

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=args_dict["cpu_threads_per_worker"]
    ) as decode_pool:
        for tar_idx, tar_path_str in enumerate(tar_paths, start=1):
            tar_path = Path(tar_path_str)
            tar_start = time.time()
            total_tars += 1
            batch: List[Tuple[str, str, bytes]] = []
            tar_image_count = 0
            try:
                for member_name, payload in iter_tar_images(tar_path):
                    batch.append((str(tar_path), member_name, payload))
                    if len(batch) >= args_dict["decode_batch_size"]:
                        rows = process_batch(
                            worker_id=worker_id,
                            gpu_id=gpu_id,
                            decode_pool=decode_pool,
                            cleanvision_scorer=cleanvision_scorer,
                            yolo_model=yolo_model,
                            yolo_device=yolo_device,
                            batch=batch,
                            min_width=args_dict["min_width"],
                            min_height=args_dict["min_height"],
                            yolo_batch_size=args_dict["yolo_batch_size"],
                            yolo_conf=args_dict["yolo_conf"],
                            yolo_iou=args_dict["yolo_iou"],
                            yolo_imgsz=args_dict["yolo_imgsz"],
                            yolo_max_det=args_dict["yolo_max_det"],
                            yolo_half=args_dict["yolo_half"],
                            head_class_ids=args_dict["head_class_ids"],
                            blur_expand_ratio=args_dict["blur_expand_ratio"],
                            blur_ratio_threshold=args_dict["blur_ratio_threshold"],
                            min_context_sharpness=args_dict["min_context_sharpness"],
                        )
                        sink.add_rows(rows)
                        total_images += len(rows)
                        total_decode_errors += sum(
                            1 for row in rows if row["decode_error"]
                        )
                        tar_image_count += len(rows)
                        batch.clear()

                if batch:
                    rows = process_batch(
                        worker_id=worker_id,
                        gpu_id=gpu_id,
                        decode_pool=decode_pool,
                        cleanvision_scorer=cleanvision_scorer,
                        yolo_model=yolo_model,
                        yolo_device=yolo_device,
                        batch=batch,
                        min_width=args_dict["min_width"],
                        min_height=args_dict["min_height"],
                        yolo_batch_size=args_dict["yolo_batch_size"],
                        yolo_conf=args_dict["yolo_conf"],
                        yolo_iou=args_dict["yolo_iou"],
                        yolo_imgsz=args_dict["yolo_imgsz"],
                        yolo_max_det=args_dict["yolo_max_det"],
                        yolo_half=args_dict["yolo_half"],
                        head_class_ids=args_dict["head_class_ids"],
                        blur_expand_ratio=args_dict["blur_expand_ratio"],
                        blur_ratio_threshold=args_dict["blur_ratio_threshold"],
                        min_context_sharpness=args_dict["min_context_sharpness"],
                    )
                    sink.add_rows(rows)
                    total_images += len(rows)
                    total_decode_errors += sum(1 for row in rows if row["decode_error"])
                    tar_image_count += len(rows)
                    batch.clear()
            except Exception as exc:  # noqa: BLE001
                log(f"{worker_name} failed on tar {tar_path}: {type(exc).__name__}: {exc}")

            elapsed = time.time() - tar_start
            log(
                f"{worker_name} tar {tar_idx}/{len(tar_paths)} done: "
                f"{tar_path.name} images={tar_image_count} elapsed={elapsed:.1f}s"
            )

    sink.close()

    worker_elapsed = time.time() - worker_start
    result = {
        "worker_id": worker_id,
        "gpu_id": gpu_id,
        "part_path": str(part_path),
        "tar_count": total_tars,
        "image_count": total_images,
        "decode_error_count": total_decode_errors,
        "elapsed_sec": worker_elapsed,
        "flush_count": sink.flush_count,
    }
    log(
        f"{worker_name} finished: images={total_images} decode_errors={total_decode_errors} "
        f"elapsed={worker_elapsed:.1f}s flushes={sink.flush_count}"
    )
    return result


def merge_parquet_parts(
    part_paths: List[Path], output_parquet: Path, compression: str
) -> None:
    if not part_paths:
        raise ValueError("No parquet parts to merge.")

    output_parquet.parent.mkdir(parents=True, exist_ok=True)
    writer: Optional[pq.ParquetWriter] = None
    try:
        for part in part_paths:
            parquet_file = pq.ParquetFile(part)
            for record_batch in parquet_file.iter_batches(batch_size=65536):
                table = pa.Table.from_batches([record_batch])
                if writer is None:
                    writer = pq.ParquetWriter(
                        where=output_parquet,
                        schema=table.schema,
                        compression=compression,
                    )
                writer.write_table(table)
    finally:
        if writer is not None:
            writer.close()


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
    parser.add_argument("--gpu-devices", type=str, default="4,5,6,7")
    parser.add_argument("--skip-yolo", action="store_true")
    parser.add_argument("--yolo-model", type=str, default="yolo11n.pt")
    parser.add_argument("--head-class-ids", type=str, default="")
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

    head_class_ids = parse_csv_ints(args.head_class_ids) if args.head_class_ids else []
    cpu_threads_per_worker = max(1, args.cpu_threads // len(gpu_ids))

    default_cv_params = ImagePropertyIssueManager().get_default_params()
    cv_thresholds = {
        "dark_threshold": (
            default_cv_params[IssueType.DARK.value]["threshold"]
            if args.dark_threshold < 0
            else args.dark_threshold
        ),
        "light_threshold": (
            default_cv_params[IssueType.LIGHT.value]["threshold"]
            if args.light_threshold < 0
            else args.light_threshold
        ),
        "low_information_threshold": (
            default_cv_params[IssueType.LOW_INFORMATION.value]["threshold"]
            if args.low_information_threshold < 0
            else args.low_information_threshold
        ),
        "low_information_normalizing_factor": (
            default_cv_params[IssueType.LOW_INFORMATION.value]["normalizing_factor"]
            if args.low_information_normalizing_factor < 0
            else args.low_information_normalizing_factor
        ),
        "blurry_threshold": (
            default_cv_params[IssueType.BLURRY.value]["threshold"]
            if args.blurry_threshold < 0
            else args.blurry_threshold
        ),
        "blurry_normalizing_factor": (
            default_cv_params[IssueType.BLURRY.value]["normalizing_factor"]
            if args.blurry_normalizing_factor < 0
            else args.blurry_normalizing_factor
        ),
        "blurry_color_threshold": (
            default_cv_params[IssueType.BLURRY.value]["color_threshold"]
            if args.blurry_color_threshold < 0
            else args.blurry_color_threshold
        ),
        "odd_size_threshold": (
            default_cv_params[IssueType.ODD_SIZE.value]["threshold"]
            if args.odd_size_threshold < 0
            else args.odd_size_threshold
        ),
    }

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
                    "yolo_model": args.yolo_model,
                    "head_class_ids": head_class_ids,
                    "yolo_batch_size": args.yolo_batch_size,
                    "yolo_conf": args.yolo_conf,
                    "yolo_iou": args.yolo_iou,
                    "yolo_imgsz": args.yolo_imgsz,
                    "yolo_max_det": args.yolo_max_det,
                    "yolo_half": bool(args.yolo_half),
                    "decode_batch_size": args.decode_batch_size,
                    "cpu_threads_per_worker": cpu_threads_per_worker,
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
        f"cpu_threads_total={args.cpu_threads}, cpu_threads_per_worker={cpu_threads_per_worker}"
    )
    log(
        f"YOLO model={args.yolo_model}, gpu_devices={gpu_ids}, "
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
        f"decode_errors={total_decode_errors} elapsed={elapsed/3600:.2f}h"
    )


if __name__ == "__main__":
    main()
