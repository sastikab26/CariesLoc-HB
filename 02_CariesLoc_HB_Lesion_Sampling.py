from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
CLASS_NAMES = {0: "primary_decay", 1: "permanent_decay"}


@dataclass(frozen=True)
class SamplingParameters:
    background_repeat: int = 2         # R_bg
    minority_weight: int = 2           # alpha
    small_weight: int = 1              # beta
    minority_small_weight: int = 1     # gamma
    multi_lesion_weight: int = 1       # delta
    primary_only_weight: int = 1      # mu
    small_area_threshold: float = 0.015  # tau_s, normalized box area
    multi_lesion_threshold: int = 3   # lambda
    max_repeat: int = 4               # R_max
    epsilon: float = 1e-12

    def validate(self) -> None:
        if not 1 <= self.background_repeat <= self.max_repeat:
            raise ValueError("background_repeat must lie in [1, max_repeat].")
        if self.max_repeat < 1 or self.multi_lesion_threshold < 1:
            raise ValueError("max_repeat and multi_lesion_threshold must be >= 1.")
        weights = (self.minority_weight, self.small_weight,
                   self.minority_small_weight, self.multi_lesion_weight,
                   self.primary_only_weight)
        if any(w < 0 for w in weights):
            raise ValueError("Sampling weights must be nonnegative.")
        if not 0 < self.small_area_threshold <= 1:
            raise ValueError("small_area_threshold must lie in (0, 1].")
        if self.epsilon <= 0:
            raise ValueError("epsilon must be positive.")


@dataclass(frozen=True)
class Annotation:
    class_id: int
    x_center: float
    y_center: float
    width: float
    height: float

    @property
    def area(self) -> float:
        return self.width * self.height


def list_images(folder: Path) -> list[Path]:
    if not folder.is_dir():
        raise NotADirectoryError(f"Image directory not found: {folder}")
    images = sorted(p.resolve() for p in folder.iterdir()
                    if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
    if not images:
        raise ValueError(f"No supported images in {folder}")
    duplicate_stems = [stem for stem, count in Counter(p.stem for p in images).items()
                       if count > 1]
    if duplicate_stems:
        raise ValueError("Ambiguous image stems (multiple image formats): "
                         + ", ".join(duplicate_stems[:10]))
    return images


def parse_yolo_annotations(label_file: Path) -> list[Annotation]:
    """Strictly validate class IDs, normalized xywh, and box containment."""
    if not label_file.is_file():
        raise FileNotFoundError(
            f"Missing label file: {label_file}. Create an empty .txt only for "
            "images verified to have no annotated lesions."
        )
    boxes: list[Annotation] = []
    for line_no, line in enumerate(label_file.read_text(encoding="utf-8-sig").splitlines(), 1):
        fields = line.split()
        if not fields:
            continue
        if len(fields) != 5:
            raise ValueError(f"{label_file}:{line_no}: expected 5 YOLO fields")
        try:
            class_value = float(fields[0])
            coords = [float(value) for value in fields[1:]]
        except ValueError as exc:
            raise ValueError(f"{label_file}:{line_no}: nonnumeric YOLO value") from exc
        if not all(math.isfinite(value) for value in [class_value, *coords]):
            raise ValueError(f"{label_file}:{line_no}: nonfinite value")
        if class_value not in CLASS_NAMES:
            raise ValueError(f"{label_file}:{line_no}: unknown class ID {class_value}")
        x, y, w, h = coords
        tolerance = 1e-6
        if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1):
            raise ValueError(f"{label_file}:{line_no}: coordinates outside YOLO range")
        if (x - w / 2 < -tolerance or x + w / 2 > 1 + tolerance or
                y - h / 2 < -tolerance or y + h / 2 > 1 + tolerance):
            raise ValueError(f"{label_file}:{line_no}: bounding box exceeds image bounds")
        boxes.append(Annotation(int(class_value), x, y, w, h))
    return boxes


def sampling_features(boxes: list[Annotation], params: SamplingParameters) -> dict:
    """Indicators defined exactly as in Algorithm 1."""
    primary = int(any(box.class_id == 0 for box in boxes))   # P_i
    minority = int(any(box.class_id == 1 for box in boxes))  # M_i
    small = int(bool(boxes) and min(box.area for box in boxes)
                < params.small_area_threshold)             # S_i
    multi = int(len(boxes) >= params.multi_lesion_threshold)  # L_i
    return {"P_i": primary, "M_i": minority, "S_i": small, "L_i": multi}


def repeat_factor(boxes: list[Annotation], params: SamplingParameters) -> tuple[int, dict]:
    """R_i* = min(R_i, R_max); background images use R_bg."""
    features = sampling_features(boxes, params)
    if not boxes:
        return params.background_repeat, features

    p, m, s, l = (features[key] for key in ("P_i", "M_i", "S_i", "L_i"))
    raw_repeat = (1 + params.minority_weight * m
                  + params.small_weight * s
                  + params.minority_small_weight * m * s
                  + params.multi_lesion_weight * l
                  + params.primary_only_weight * p * (1 - m))
    return min(raw_repeat, params.max_repeat), features


def _write_dataset_yaml(path: Path, train_list: Path, validation_images: Path,
                        test_images: Path | None) -> None:
    # Simple quoted YAML via JSON string representation (JSON strings are valid YAML).
    lines = [f"train: {json.dumps(str(train_list.resolve()))}",
             f"val: {json.dumps(str(validation_images.resolve()))}"]
    if test_images is not None:
        lines.append(f"test: {json.dumps(str(test_images.resolve()))}")
    lines.extend(["names:", "  0: primary_decay", "  1: permanent_decay", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def sample_training_dataset(train_images: Path, train_labels: Path,
                            val_images: Path, output_dir: Path,
                            test_images: Path | None = None,
                            parameters: SamplingParameters | None = None) -> dict:
    """Generate weighted training-only manifest plus reproducibility reports."""
    params = parameters or SamplingParameters()
    params.validate()
    train_images, train_labels, val_images, output_dir = (
        p.expanduser().resolve() for p in (train_images, train_labels, val_images, output_dir)
    )
    test_images = test_images.expanduser().resolve() if test_images is not None else None
    if not train_labels.is_dir() or not val_images.is_dir():
        raise NotADirectoryError("The training-label and validation-image directories must exist.")
    if test_images is not None and not test_images.is_dir():
        raise NotADirectoryError(f"Test images directory not found: {test_images}")
    images = list_images(train_images)
    if output_dir == train_images or output_dir == train_labels:
        raise ValueError("Output directory must be different from source directories.")

    weighted_paths: list[str] = []
    audit_rows: list[dict] = []
    original_class_count: Counter[int] = Counter()
    weighted_class_count: Counter[int] = Counter()
    repeat_distribution: Counter[int] = Counter()
    background_images = 0

    # Validate and build the complete plan before writing any output files.
    for image in images:
        boxes = parse_yolo_annotations(train_labels / f"{image.stem}.txt")
        repeats, flags = repeat_factor(boxes, params)
        if not boxes:
            background_images += 1
        repeat_distribution[repeats] += 1
        for box in boxes:
            original_class_count[box.class_id] += 1
            weighted_class_count[box.class_id] += repeats
        weighted_paths.extend([str(image)] * repeats)
        audit_rows.append({
            "image_name": image.name,
            "background": int(not boxes),
            "num_boxes": len(boxes),
            "P_i": flags["P_i"],
            "M_i": flags["M_i"],
            "S_i": flags["S_i"],
            "L_i": flags["L_i"],
            "minimum_normalized_box_area": min((b.area for b in boxes), default=""),
            "repeat_factor": repeats,
        })

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = output_dir / "weighted_train.txt"
    manifest.write_text("\n".join(weighted_paths) + "\n", encoding="utf-8")

    with (output_dir / "sampling_audit.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(audit_rows[0]))
        writer.writeheader()
        writer.writerows(audit_rows)

    original_primary = original_class_count[0]
    original_permanent = original_class_count[1]
    weighted_primary = weighted_class_count[0]
    weighted_permanent = weighted_class_count[1]
    summary = {
        "method": "CariesLoc-HB Algorithm 1: hard-background enhanced lesion-aware sampling",
        "class_names": {str(key): name for key, name in CLASS_NAMES.items()},
        "parameters": asdict(params),
        "original_train_images": len(images),
        "background_train_images": background_images,
        "positive_train_images": len(images) - background_images,
        "weighted_train_samples": len(weighted_paths),
        "original_box_counts": {"primary_decay": original_primary,
                                "permanent_decay": original_permanent},
        "weighted_box_counts": {"primary_decay": weighted_primary,
                                "permanent_decay": weighted_permanent},
        "weighted_class_imbalance_ratio": (
            weighted_primary / max(weighted_permanent, params.epsilon)
        ),
        "repeat_factor_distribution": {str(key): value for key, value in sorted(repeat_distribution.items())},
        "note": "Weighted class counts refer to sampled bounding-box occurrences, not unique annotations.",
    }
    (output_dir / "sampling_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    _write_dataset_yaml(output_dir / "sampling_data.yaml", manifest, val_images, test_images)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train-images", required=True, type=Path,
                        help="Existing training image directory")
    parser.add_argument("--train-labels", required=True, type=Path,
                        help="Existing, validated training YOLO labels directory")
    parser.add_argument("--val-images", required=True, type=Path,
                        help="Unmodified validation image directory")
    parser.add_argument("--test-images", type=Path,
                        help="Optional unmodified test image directory")
    parser.add_argument("--output-dir", required=True, type=Path,
                        help="Directory for weighted manifest and reports")
    args = parser.parse_args()
    summary = sample_training_dataset(args.train_images, args.train_labels,
                                      args.val_images, args.output_dir, args.test_images)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
