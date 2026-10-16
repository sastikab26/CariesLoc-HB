from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger("cariesloc_hb.evaluation")
EXPECTED_NAMES = ("primary_decay", "permanent_decay")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", type=Path, required=True, help="Frozen .pt checkpoint")
    parser.add_argument("--data", type=Path, required=True, help="Ultralytics dataset YAML")
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--out", type=Path, required=True, help="Directory for reports and plots")
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--conf", type=float, default=0.001, help="Minimum prediction confidence used for evaluation")
    parser.add_argument("--nms-iou", type=float, default=0.50, help="IoU for NMS, not the AP matching threshold")
    parser.add_argument("--device", default="0", help="GPU index, 'cpu', or supported Ultralytics device")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--save-json", action="store_true", help="Write predicted detections in JSON when supported")
    parser.add_argument("--dataset-label", default="test", help="Descriptive dataset label for the report")
    parser.add_argument("--allow-other-classes", action="store_true", help="Disable the expected two-class name check")
    parser.add_argument("--overwrite", action="store_true", help="Allow writing into an existing nonempty result directory")
    return parser.parse_args()


def validate_inputs(args: argparse.Namespace) -> None:
    for name in ("weights", "data"):
        path = getattr(args, name)
        if not path.is_file():
            raise FileNotFoundError(f"{name}: file not found: {path}")
    if args.imgsz <= 0 or args.batch <= 0 or args.workers < 0 or args.max_det < 1:
        raise ValueError("imgsz, batch and max-det must be positive; workers must be nonnegative")
    if not (0.0 <= args.conf <= 1.0) or not (0.0 < args.nms_iou <= 1.0):
        raise ValueError("conf must be within [0,1] and nms-iou within (0,1]")
    if args.out.exists() and any(args.out.iterdir()) and not args.overwrite:
        raise FileExistsError("Output directory is not empty; supply a new --out or --overwrite")


def normalize_names(names: Any) -> dict[int, str]:
    if isinstance(names, (tuple, list)):
        return {idx: str(value) for idx, value in enumerate(names)}
    if isinstance(names, dict):
        return {int(key): str(value) for key, value in names.items()}
    raise TypeError(f"Unsupported class-name format: {type(names).__name__}")


def safe_float(value: Any) -> float | None:
    if value is None:
        return None
    x = float(value)
    return x if math.isfinite(x) else None


def f1_from_pr(precision: float | None, recall: float | None) -> float | None:
    if precision is None or recall is None or precision + recall <= 0:
        return None
    return 2.0 * precision * recall / (precision + recall)


def row_class_metrics(metrics: Any, names: dict[int, str]) -> list[dict[str, Any]]:
    """Map AP array positions to real class IDs via ap_class_index.

    Ultralytics `box.p`, `box.r`, `box.ap50` and `box.ap` are indexed by
    classes occurring in evaluation targets, *not* necessarily directly
    by the underlying 0..N-1 class ID. Missing target classes are NA.
    """
    box = metrics.box
    class_ids = [int(c) for c in box.ap_class_index]
    if len(class_ids) != len(box.ap50):
        raise RuntimeError("Per-class AP indices and values have inconsistent lengths")
    index_by_class = {cid: ix for ix, cid in enumerate(class_ids)}
    rows = []
    for class_id, class_name in sorted(names.items()):
        idx = index_by_class.get(class_id)
        if idx is None:
            rows.append({"class_id": class_id, "class": class_name, "precision": None,
                         "recall": None, "f1": None, "ap50": None, "ap50_95": None,
                         "status": "not_present_in_ground_truth"})
            continue
        p = safe_float(box.p[idx])
        r = safe_float(box.r[idx])
        rows.append({"class_id": class_id, "class": class_name,
                     "precision": p, "recall": r, "f1": f1_from_pr(p, r),
                     "ap50": safe_float(box.ap50[idx]),
                     "ap50_95": safe_float(box.ap[idx]), "status": "evaluated"})
    return rows


def row_overall_metrics(metrics: Any) -> dict[str, Any]:
    box = metrics.box
    p, r = safe_float(box.mp), safe_float(box.mr)
    return {"class": "overall", "precision": p, "recall": r,
            "f1": f1_from_pr(p, r), "map50": safe_float(box.map50),
            "map50_95": safe_float(box.map)}


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def run() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    validate_inputs(args)

    try:
        import ultralytics
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit("Install requirements first: pip install ultralytics") from exc

    model = YOLO(str(args.weights.resolve()))
    model_names = normalize_names(model.names)
    if not args.allow_other_classes and tuple(model_names.get(i) for i in range(2)) != EXPECTED_NAMES:
        raise ValueError(f"Expected class order {EXPECTED_NAMES}; checkpoint contains {model_names}. "
                         "Check class mapping or use --allow-other-classes.")

    args.out.mkdir(parents=True, exist_ok=True)
    LOGGER.info("Evaluating %s on %s split", args.dataset_label, args.split)
    # `model.val` applies standard detection preprocessing, including letterboxing.
    # No augmentation, retraining, test-time augmentation or tuning is performed.
    metrics = model.val(
        data=str(args.data.resolve()), split=args.split, imgsz=args.imgsz,
        batch=args.batch, conf=args.conf, iou=args.nms_iou,
        max_det=args.max_det, device=args.device, workers=args.workers,
        plots=True, save_json=args.save_json, augment=False,
        project=str(args.out.resolve()), name="ultralytics_validation", exist_ok=False,
        verbose=True,
    )

    overall = row_overall_metrics(metrics)
    class_rows = row_class_metrics(metrics, model_names)
    metadata = {
        "evaluation_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_label": args.dataset_label,
        "split": args.split,
        "model_class_names": model_names,
        "image_size": args.imgsz,
        "batch_size": args.batch,
        "confidence_floor": args.conf,
        "nms_iou": args.nms_iou,
        "max_detections": args.max_det,
        "confidence_note": (
            "AP@0.5 and AP@0.5:0.95 integrate precision/recall across confidence; "
            "a high confidence floor truncates that curve."
        ),
        "precision_recall_note": (
            "Ultralytics per-class precision/recall are at its curve-selected operating point; "
            "F1 is derived from these rounded-unmodified values."
        ),
        "ap_note": "Per-class AP is not mAP. mAP averages AP across evaluated classes.",
        "ultralytics_version": ultralytics.__version__,
        "python_version": platform.python_version(),
        "weights_filename": args.weights.name,
        "data_filename": args.data.name,
        "retraining": False,
        "test_time_augmentation": False,
    }
    (args.out / "evaluation_results.json").write_text(
        json.dumps({"metadata": metadata, "overall": overall, "per_class": class_rows},
                   indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    write_csv(args.out / "overall_metrics.csv", [overall],
              ["class", "precision", "recall", "f1", "map50", "map50_95"])
    write_csv(args.out / "per_class_metrics.csv", class_rows,
              ["class_id", "class", "precision", "recall", "f1", "ap50", "ap50_95", "status"])
    LOGGER.info("Saved summary tables and metadata to %s", args.out)
    LOGGER.info("Overall metrics: %s", overall)
    for row in class_rows:
        LOGGER.info("Per-class metrics: %s", row)


if __name__ == "__main__":
    try:
        run()
    except (ValueError, FileNotFoundError, FileExistsError, RuntimeError) as exc:
        logging.error("Evaluation failed: %s", exc)
        sys.exit(1)
