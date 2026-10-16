from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class LesionPreservingAugmentation:
    """Exact detection-augmentation settings from the supplied training code.

    YOLO interprets ``scale`` as the magnitude of symmetric scaling (0.18
    implies approximately 0.82–1.18), and ``translate`` as a relative fraction.
    ``close_mosaic`` is the number of *final* epochs with mosaic disabled.
    """

    hsv_h: float = 0.010
    hsv_s: float = 0.35
    hsv_v: float = 0.22
    degrees: float = 2.5
    translate: float = 0.045
    scale: float = 0.18
    shear: float = 0.0
    perspective: float = 0.00015
    fliplr: float = 0.5
    flipud: float = 0.0
    mosaic: float = 0.15
    close_mosaic: int = 40
    mixup: float = 0.0
    copy_paste: float = 0.0
    erasing: float = 0.05

    def validate(self) -> None:
        values = asdict(self)
        for name, value in values.items():
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        for name in ("fliplr", "flipud", "mosaic", "mixup", "copy_paste", "erasing"):
            if not 0.0 <= values[name] <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        for name in ("hsv_h", "hsv_s", "hsv_v", "translate", "scale", "shear", "perspective"):
            if values[name] < 0.0:
                raise ValueError(f"{name} must be nonnegative")
        if not 0.0 <= self.hsv_h <= 0.5:
            raise ValueError("hsv_h must be <= 0.5")
        if self.scale >= 1.0:
            raise ValueError("scale must be below 1.0 to avoid zero/negative scale")
        if self.close_mosaic < 0:
            raise ValueError("close_mosaic must be nonnegative")
        if self.flipud != 0.0 or self.mixup != 0.0 or self.copy_paste != 0.0:
            raise ValueError("Vertical flip, MixUp and Copy-Paste must remain disabled")

    def to_ultralytics_kwargs(self) -> dict[str, float | int]:
        """Return keyword arguments for ``ultralytics.YOLO.train``.

        Bounding boxes are transformed automatically with the images by the
        detection dataloader. This function does not modify data in place.
        """
        self.validate()
        return asdict(self)


def detection_augmentation_kwargs() -> dict[str, float | int]:
    """Augmentation-only arguments to unpack into a separate training call."""
    return LesionPreservingAugmentation().to_ultralytics_kwargs()


def export_configuration(output_path: str | Path) -> Path:
    """Save parameter values for a reproducible GitHub release."""
    output = Path(output_path).expanduser()
    params = detection_augmentation_kwargs()
    if output.suffix.lower() == ".json":
        serialized = json.dumps(params, indent=2, sort_keys=True) + "\n"
    elif output.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as exc:
            raise SystemExit("PyYAML is required for YAML output") from exc
        serialized = yaml.safe_dump(params, sort_keys=True)
    else:
        raise ValueError("Output extension must be .json, .yaml, or .yml")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(serialized, encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export CariesLoc-HB training-time augmentation settings."
    )
    parser.add_argument(
        "--output", type=Path, help="Optional JSON or YAML configuration file."
    )
    args = parser.parse_args()
    if args.output is None:
        print(json.dumps(detection_augmentation_kwargs(), indent=2, sort_keys=True))
    else:
        print(export_configuration(args.output))


if __name__ == "__main__":
    main()
