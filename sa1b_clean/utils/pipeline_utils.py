from __future__ import annotations

import concurrent.futures
import hashlib
import io
import json
import math
import tarfile
import time
from dataclasses import asdict, dataclass
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

FALLBACK_CV_DEFAULTS: Dict[str, Dict[str, float]] = {
    IssueType.DARK.value: {"threshold": 0.32},
    IssueType.LIGHT.value: {"threshold": 0.05},
    IssueType.LOW_INFORMATION.value: {"threshold": 0.3, "normalizing_factor": 0.1},
    IssueType.BLURRY.value: {
        "threshold": 0.29,
        "normalizing_factor": 0.01,
        "color_threshold": 0.18,
    },
    IssueType.ODD_SIZE.value: {"threshold": 10.0},
}


# Default model path inside this repository:
# /<repo>/models/medium.pt
DEFAULT_LOCAL_HEAD_MODEL = str((Path(__file__).resolve().parents[2] / "models" / "medium.pt"))

# Common COCO checkpoints (not head-specific).
DISALLOWED_COCO_CHECKPOINTS = {
    "yolov8n.pt",
    "yolov8s.pt",
    "yolov8m.pt",
    "yolov8l.pt",
    "yolov8x.pt",
    "yolo11n.pt",
    "yolo11s.pt",
    "yolo11m.pt",
    "yolo11l.pt",
    "yolo11x.pt",
}


def ts() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def log(msg: str) -> None:
    print(f"[{ts()}] {msg}", flush=True)


def parse_csv_ints(csv_text: str) -> List[int]:
    return [int(x.strip()) for x in csv_text.split(",") if x.strip()]


def list_tar_files(root: Path, recursive: bool) -> List[Path]:
    pattern = "**/*.tar" if recursive else "*.tar"
    return sorted(root.glob(pattern))


def split_round_robin(items: Sequence[Path], bucket_count: int) -> List[List[Path]]:
    buckets: List[List[Path]] = [[] for _ in range(bucket_count)]
    for idx, item in enumerate(items):
        buckets[idx % bucket_count].append(item)
    return buckets


def ensure_local_yolo_model(model_ref: str) -> str:
    model_path = Path(model_ref).expanduser()
    if not model_path.is_absolute():
        model_path = (Path.cwd() / model_path).resolve()
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found: {model_path}. "
            f"Please place YOLOv8 head model at {DEFAULT_LOCAL_HEAD_MODEL} "
            "or pass --yolo-model /absolute/path/to/model.pt."
        )
    return str(model_path)


def validate_head_model(yolo_model: str, skip_yolo: bool) -> None:
    if skip_yolo:
        return
    model_name = Path(yolo_model.strip()).name.lower()
    if model_name in DISALLOWED_COCO_CHECKPOINTS:
        raise ValueError(
            f"{model_name} is a common COCO checkpoint, not a dedicated head detector. "
            "This pipeline defaults to YOLOv8_head_detector medium.pt. "
            "Please pass a dedicated head model path."
        )


def resolve_cv_param(
    default_cv_params: Dict[str, Dict[str, Any]],
    issue_type: str,
    key: str,
    arg_value: float,
) -> float:
    if arg_value >= 0:
        return float(arg_value)

    issue_params = default_cv_params.get(issue_type, {})
    if key in issue_params:
        return float(issue_params[key])

    fallback_issue_params = FALLBACK_CV_DEFAULTS.get(issue_type, {})
    if key in fallback_issue_params:
        value = float(fallback_issue_params[key])
        log(f"CleanVision default missing {issue_type}.{key}; using fallback {value}")
        return value

    raise KeyError(f"Missing CleanVision parameter: {issue_type}.{key}")


def build_cv_thresholds(
    dark_threshold: float,
    light_threshold: float,
    low_information_threshold: float,
    low_information_normalizing_factor: float,
    blurry_threshold: float,
    blurry_normalizing_factor: float,
    blurry_color_threshold: float,
    odd_size_threshold: float,
) -> Dict[str, float]:
    default_cv_params = ImagePropertyIssueManager().get_default_params()
    return {
        "dark_threshold": resolve_cv_param(
            default_cv_params=default_cv_params,
            issue_type=IssueType.DARK.value,
            key="threshold",
            arg_value=dark_threshold,
        ),
        "light_threshold": resolve_cv_param(
            default_cv_params=default_cv_params,
            issue_type=IssueType.LIGHT.value,
            key="threshold",
            arg_value=light_threshold,
        ),
        "low_information_threshold": resolve_cv_param(
            default_cv_params=default_cv_params,
            issue_type=IssueType.LOW_INFORMATION.value,
            key="threshold",
            arg_value=low_information_threshold,
        ),
        "low_information_normalizing_factor": resolve_cv_param(
            default_cv_params=default_cv_params,
            issue_type=IssueType.LOW_INFORMATION.value,
            key="normalizing_factor",
            arg_value=low_information_normalizing_factor,
        ),
        "blurry_threshold": resolve_cv_param(
            default_cv_params=default_cv_params,
            issue_type=IssueType.BLURRY.value,
            key="threshold",
            arg_value=blurry_threshold,
        ),
        "blurry_normalizing_factor": resolve_cv_param(
            default_cv_params=default_cv_params,
            issue_type=IssueType.BLURRY.value,
            key="normalizing_factor",
            arg_value=blurry_normalizing_factor,
        ),
        "blurry_color_threshold": resolve_cv_param(
            default_cv_params=default_cv_params,
            issue_type=IssueType.BLURRY.value,
            key="color_threshold",
            arg_value=blurry_color_threshold,
        ),
        "odd_size_threshold": resolve_cv_param(
            default_cv_params=default_cv_params,
            issue_type=IssueType.ODD_SIZE.value,
            key="threshold",
            arg_value=odd_size_threshold,
        ),
    }


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


def _score_cv_chunk(
    pil_images_chunk: List[Image.Image], thresholds_dict: Dict[str, float]
) -> List[Dict[str, Any]]:
    scorer = CleanVisionBatchScorer(CVThresholds(**thresholds_dict))
    return scorer.score_batch(pil_images_chunk)


def score_cleanvision_metrics(
    pil_images: List[Image.Image],
    cleanvision_scorer: CleanVisionBatchScorer,
    cv_workers: int,
    cv_chunk_size: int,
) -> List[Dict[str, Any]]:
    if not pil_images:
        return []

    cv_workers = max(1, int(cv_workers))
    cv_chunk_size = max(1, int(cv_chunk_size))

    if cv_workers == 1 or len(pil_images) <= cv_chunk_size:
        return cleanvision_scorer.score_batch(pil_images)

    chunks = [
        pil_images[i : i + cv_chunk_size]
        for i in range(0, len(pil_images), cv_chunk_size)
    ]
    if len(chunks) == 1:
        return cleanvision_scorer.score_batch(pil_images)

    thresholds_dict = asdict(cleanvision_scorer.thresholds)
    def _run_chunk(chunk: List[Image.Image]) -> List[Dict[str, Any]]:
        return _score_cv_chunk(chunk, thresholds_dict)

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=min(cv_workers, len(chunks))
    ) as pool:
        chunk_outputs = list(pool.map(_run_chunk, chunks))

    merged: List[Dict[str, Any]] = []
    for output in chunk_outputs:
        merged.extend(output)
    return merged


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
    cv_workers: int,
    cv_chunk_size: int,
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

    cv_metrics = score_cleanvision_metrics(
        pil_images=pil_images,
        cleanvision_scorer=cleanvision_scorer,
        cv_workers=cv_workers,
        cv_chunk_size=cv_chunk_size,
    )

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
    cv_workers = max(1, min(args_dict["cv_workers"], args_dict["cpu_threads_per_worker"]))
    cv_chunk_size = max(1, args_dict["cv_chunk_size"])

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
                            cv_workers=cv_workers,
                            cv_chunk_size=cv_chunk_size,
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
                        cv_workers=cv_workers,
                        cv_chunk_size=cv_chunk_size,
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
