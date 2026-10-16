from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import math
import shutil
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

from PIL import Image, UnidentifiedImageError

LOGGER = logging.getLogger("cariesloc_hb.preprocessing")
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CLASS_NAMES = {0: "primary_decay", 1: "permanent_decay"}
SPLIT_ALIASES = {"train": ("train",), "val": ("val", "valid"), "test": ("test",)}


@dataclass(frozen=True)
class Box:
    cls: int
    x: float
    y: float
    w: float
    h: float

    def yolo_line(self) -> str:
        return f"{self.cls} {self.x:.10f} {self.y:.10f} {self.w:.10f} {self.h:.10f}"


@dataclass(frozen=True)
class Record:
    split: str
    image: str
    source_width: int
    source_height: int
    output_width: int
    output_height: int
    num_boxes: int
    primary_decay_boxes: int
    permanent_decay_boxes: int
    is_background: bool
    missing_label_accepted: bool
    rejected_boxes: int


class AnnotationError(ValueError):
    """Invalid or ambiguously annotated input image."""


def parse_yolo_labels(path: Path, tolerance: float = 1e-6) -> tuple[list[Box], int]:
    """Keep strictly valid normalized boxes; count malformed lines.

    No synthetic corrections are made to clinical annotations. Tiny floating
    point overshoots up to ``tolerance`` are accepted and clamped to [0, 1].
    """
    accepted: list[Box] = []
    rejected = 0
    for line_number, raw in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        stripped = raw.strip()
        if not stripped:
            continue
        fields = stripped.split()
        if len(fields) != 5:
            LOGGER.warning("Invalid label row at %s:%d (expected five fields)", path, line_number)
            rejected += 1
            continue
        try:
            # Require class id to be an integer token, avoiding e.g. 0.5 -> 0.
            cls = int(fields[0])
            coords = [float(value) for value in fields[1:]]
        except ValueError:
            rejected += 1
            continue
        if cls not in CLASS_NAMES or not all(math.isfinite(v) for v in coords):
            rejected += 1
            continue
        x, y, w, h = coords
        if (w <= 0 or h <= 0 or any(v < -tolerance or v > 1 + tolerance for v in coords)
                or x - w / 2 < -tolerance or y - h / 2 < -tolerance
                or x + w / 2 > 1 + tolerance or y + h / 2 > 1 + tolerance):
            rejected += 1
            continue
        # Clamp only rounding-scale deviations, never materially incorrect boxes.
        x1 = max(0.0, x - w / 2)
        y1 = max(0.0, y - h / 2)
        x2 = min(1.0, x + w / 2)
        y2 = min(1.0, y + h / 2)
        accepted.append(Box(cls, (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1))
    return accepted, rejected


def letterbox(image: Image.Image, boxes: list[Box], size: int) -> tuple[Image.Image, list[Box]]:
    """Apply aspect-preserving letterbox and the same geometry to YOLO boxes.

    Coordinates are computed from the actual integer resized dimensions and
    canvas offsets, not an idealized floating-point scale.
    """
    src_w, src_h = image.size
    ratio = min(size / src_w, size / src_h)
    dst_w = min(size, max(1, round(src_w * ratio)))
    dst_h = min(size, max(1, round(src_h * ratio)))
    left = (size - dst_w) // 2
    top = (size - dst_h) // 2
    resized = image.resize((dst_w, dst_h), Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", (size, size), (114, 114, 114))
    canvas.paste(resized, (left, top))

    updated: list[Box] = []
    for box in boxes:
        # Source coordinates are normalized; transform through resized pixels.
        updated.append(Box(
            cls=box.cls,
            x=(box.x * dst_w + left) / size,
            y=(box.y * dst_h + top) / size,
            w=box.w * dst_w / size,
            h=box.h * dst_h / size,
        ))
    return canvas, updated


def image_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def identify_splits(root: Path) -> dict[str, tuple[Path, Path]]:
    splits: dict[str, tuple[Path, Path]] = {}
    for canonical, options in SPLIT_ALIASES.items():
        for name in options:
            img_dir = root / "images" / name
            label_dir = root / "labels" / name
            if img_dir.is_dir():
                if not label_dir.is_dir():
                    raise FileNotFoundError(f"Missing labels directory: {label_dir}")
                splits[canonical] = (img_dir, label_dir)
                break
    if not splits:
        raise FileNotFoundError("No images/{train,val,test} directories found")
    return splits


def list_images(directory: Path) -> list[Path]:
    images = sorted((p for p in directory.iterdir() if p.is_file()
                     and p.suffix.lower() in SUPPORTED_EXTENSIONS), key=lambda p: p.name.lower())
    stems = [p.stem.lower() for p in images]
    if len(stems) != len(set(stems)):
        raise AnnotationError(f"Duplicate image stems with different extensions in {directory}")
    return images


def process_dataset(
    input_root: Path,
    output_root: Path,
    *,
    letterbox_size: Optional[int] = None,
    missing_label_is_background: bool = False,
    invalid_policy: str = "error",
    check_duplicate_images: bool = True,
) -> dict:
    """Validate and export images/labels without mixing patient-wise splits.

    ``invalid_policy='error'`` is the recommended research setting: fail when
    any invalid annotation is found. ``drop-box`` is an explicitly auditable
    alternative for already reviewed datasets, but it will NOT silently turn
    an all-invalid positive into a background example.
    """
    input_root = input_root.expanduser().resolve()
    output_root = output_root.expanduser().resolve()
    if input_root == output_root or input_root in output_root.parents or output_root in input_root.parents:
        raise ValueError("Input and output must be separate non-nested directories")
    if letterbox_size is not None and letterbox_size <= 0:
        raise ValueError("--letterbox-size must be positive")
    if invalid_policy not in {"error", "drop-box"}:
        raise ValueError("invalid_policy must be 'error' or 'drop-box'")
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError("Output directory must be new or empty to avoid stale labels")

    splits = identify_splits(input_root)
    records: list[Record] = []
    hashes: dict[str, str] = {}
    statistics: dict[str, dict] = {}

    for split, (image_dir, label_dir) in splits.items():
        files = list_images(image_dir)
        if not files:
            raise AnnotationError(f"No images found in {image_dir}")
        expected_label_names = {p.stem.lower() + ".txt" for p in files}
        orphans = sorted(p.name for p in label_dir.glob("*.txt") if p.name.lower() not in expected_label_names)
        if orphans:
            raise AnnotationError(f"Orphan labels in {label_dir}: {orphans[:10]}")
        counts: Counter = Counter()

        for image_path in files:
            label_path = label_dir / f"{image_path.stem}.txt"
            label_missing = not label_path.is_file()
            if label_missing and not missing_label_is_background:
                raise AnnotationError(f"Missing annotation for {image_path.name} in {split}")
            boxes, rejected = ([], 0) if label_missing else parse_yolo_labels(label_path)
            if rejected and (invalid_policy == "error" or not boxes):
                raise AnnotationError(
                    f"{image_path.name} has {rejected} invalid boxes; "
                    "review source annotations before exporting"
                )

            try:
                with Image.open(image_path) as opened:
                    if opened.getexif().get(274, 1) != 1:
                        raise AnnotationError(
                            f"EXIF orientation in {image_path.name} must be resolved together "
                            "with ground-truth annotations before preprocessing"
                        )
                    opened.load()
                    source_is_rgb = opened.mode == "RGB"
                    image = opened.convert("RGB")
            except (UnidentifiedImageError, OSError) as exc:
                raise AnnotationError(f"Unreadable image: {image_path}") from exc

            width, height = image.size
            if width < 2 or height < 2:
                raise AnnotationError(f"Image too small: {image_path}")

            if check_duplicate_images:
                digest = image_sha256(image_path)
                previous_split = hashes.get(digest)
                if previous_split is not None and previous_split != split:
                    raise AnnotationError(
                        f"Identical image bytes appear in {previous_split} and {split}: "
                        f"{image_path.name} (review split leakage)"
                    )
                hashes[digest] = split

            exported_image = image
            exported_boxes = boxes
            if letterbox_size is not None:
                exported_image, exported_boxes = letterbox(image, boxes, letterbox_size)

            target_images = output_root / "images" / split
            target_labels = output_root / "labels" / split
            target_images.mkdir(parents=True, exist_ok=True)
            target_labels.mkdir(parents=True, exist_ok=True)

            if letterbox_size is None and source_is_rgb and image_path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                # No pixel changes in copy mode. Original image bytes retained.
                output_image_path = target_images / image_path.name
                shutil.copyfile(image_path, output_image_path)
            else:
                # Lossless RGB PNG for explicit transformations or unsupported inputs.
                output_image_path = target_images / f"{image_path.stem}.png"
                exported_image.save(output_image_path, format="PNG")
            output_label_path = target_labels / f"{image_path.stem}.txt"
            content = "\n".join(b.yolo_line() for b in exported_boxes)
            output_label_path.write_text(content + ("\n" if content else ""), encoding="utf-8")

            count_primary = sum(b.cls == 0 for b in exported_boxes)
            count_permanent = sum(b.cls == 1 for b in exported_boxes)
            counts["images"] += 1
            counts["background_images"] += int(not exported_boxes)
            counts["positive_images"] += int(bool(exported_boxes))
            counts["primary_decay_boxes"] += count_primary
            counts["permanent_decay_boxes"] += count_permanent
            counts["invalid_boxes_dropped"] += rejected
            counts["missing_labels_accepted"] += int(label_missing)
            records.append(Record(
                split=split, image=output_image_path.name,
                source_width=width, source_height=height,
                output_width=exported_image.width, output_height=exported_image.height,
                num_boxes=len(exported_boxes), primary_decay_boxes=count_primary,
                permanent_decay_boxes=count_permanent,
                is_background=not exported_boxes,
                missing_label_accepted=label_missing,
                rejected_boxes=rejected,
            ))
        statistics[split] = dict(counts)
        LOGGER.info("%s: %s", split, statistics[split])

    audit_dir = output_root / "audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    with (audit_dir / "preprocessing_manifest.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(Record.__dataclass_fields__))
        writer.writeheader()
        writer.writerows(asdict(record) for record in records)
    report = {
        "classes": CLASS_NAMES,
        "letterbox_size": letterbox_size,
        "normalization": "YOLO xywh, normalized to exported image dimensions",
        "color_processing": "none; RGB conversion only when required",
        "photometric_enhancement": "none",
        "splits_preserved": True,
        "patient_split_verified": False,
        "same_image_duplicate_check": check_duplicate_images,
        "missing_label_is_background": missing_label_is_background,
        "invalid_annotation_policy": invalid_policy,
        "statistics": statistics,
    }
    (audit_dir / "preprocessing_summary.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-root", type=Path, required=True, help="Source patient-split YOLO dataset root")
    parser.add_argument("--output-root", type=Path, required=True, help="New output directory")
    parser.add_argument("--letterbox-size", type=int, default=None,
                        help="Optional offline square letterboxing; omit to retain image pixels")
    parser.add_argument("--missing-label-is-background", action="store_true",
                        help="Treat missing label as negative ONLY if source annotation protocol guarantees this")
    parser.add_argument("--invalid-policy", choices=("error", "drop-box"), default="error",
                        help="Fail on malformed labels (default), or drop invalid boxes with audit counts")
    parser.add_argument("--skip-image-hash-check", action="store_true",
                        help="Disable same-byte cross-split leakage detection")
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = build_parser().parse_args()
    process_dataset(
        args.input_root, args.output_root,
        letterbox_size=args.letterbox_size,
        missing_label_is_background=args.missing_label_is_background,
        invalid_policy=args.invalid_policy,
        check_duplicate_images=not args.skip_image_hash_check,
    )
    LOGGER.info("Preprocessing completed. See output audit/ for the export report.")


if __name__ == "__main__":
    main()
