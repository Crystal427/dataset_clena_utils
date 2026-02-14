# Agent Notes

## Code Structure Rule
- `sa1b_clean/sa1b_tar_pipeline.py` is the CLI entrypoint only.
- Reusable logic must live in `sa1b_clean/utils/` and be imported by the entrypoint.
- When adding new functionality, prefer extending `sa1b_clean/utils/pipeline_utils.py` first.

## YOLO Head Model Rule
- The pipeline is configured for **head detection**.
- Do not use common COCO checkpoints (for example `yolov8n.pt`, `yolov8m.pt`, `yolo11n.pt`) for head detection.
- Use one of these options:
  - Local head model file via `--yolo-model /path/to/head_model.pt`
- Default local model path is `<repo>/models/medium.pt` (committed file).

## Operational Defaults
- GPU devices default: `4,5,6,7`
- CPU threads default: `220`
- Writer mode default: `buffered` (recommended for SA-1B scale)
