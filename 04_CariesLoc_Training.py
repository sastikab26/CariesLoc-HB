from __future__ import annotations

import argparse
import json
import logging
import platform
import random
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from importlib.util import module_from_spec, spec_from_file_location


def detection_augmentation_kwargs():
    """Load augmentation settings from the numbered companion script."""
    module_path = Path(__file__).resolve().with_name("03_CariesLoc_Lesion_Augmentation.py")
    spec = spec_from_file_location("cariesloc_augmentation", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load augmentation module: {module_path}")
    import sys
    module = module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.detection_augmentation_kwargs()

LOGGER = logging.getLogger("cariesloc_hb.training")


@dataclass(frozen=True)
class TrainingConfig:
    """Parameters reported for the CariesLoc-HB YOLO11m experiment."""

    model: str = "yolo11m.pt"
    imgsz: int = 1024
    batch: int = 4
    epochs: int = 180
    patience: int = 45
    optimizer: str = "AdamW"
    lr0: float = 0.00030
    lrf: float = 0.01
    weight_decay: float = 0.0005
    momentum: float = 0.937
    cos_lr: bool = True
    amp: bool = True
    seed: int = 42
    deterministic: bool = True
    pretrained: bool = True
    workers: int = 0
    box: float = 8.5
    cls: float = 0.65
    dfl: float = 1.7
    save_period: int = 1
    val: bool = True
    plots: bool = True

    def validate(self) -> None:
        if self.imgsz <= 0 or self.batch <= 0 or self.epochs <= 0:
            raise ValueError("imgsz, batch, and epochs must be positive")
        if self.patience < 0 or self.workers < 0:
            raise ValueError("patience and workers must be nonnegative")
        if not (0.0 < self.lr0 and 0.0 < self.lrf <= 1.0):
            raise ValueError("learning-rate values are invalid")
        if self.save_period < 1:
            raise ValueError("save_period must be positive")


def validate_dataset_yaml(dataset_yaml: Path) -> dict[str, Any]:
    """Check file shape, training manifest, and unweighted validation source.

    This is a configuration check, not a replacement for preprocessing or
    annotation verification. The sampling stage handles those checks.
    """
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required: pip install pyyaml") from exc

    if not dataset_yaml.is_file():
        raise FileNotFoundError(f"Dataset descriptor not found: {dataset_yaml}")
    data = yaml.safe_load(dataset_yaml.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Dataset YAML must contain a mapping")
    for key in ("train", "val", "names"):
        if key not in data:
            raise ValueError(f"Dataset YAML is missing required key: {key}")
    names = data["names"]
    expected = {0: "primary_decay", 1: "permanent_decay"}
    if isinstance(names, list):
        actual = dict(enumerate(names))
    elif isinstance(names, dict):
        actual = {int(key): value for key, value in names.items()}
    else:
        raise ValueError("names must be a list or mapping")
    if actual != expected:
        raise ValueError(f"Expected class IDs {expected}; received {actual}")

    train = data["train"]
    if not isinstance(train, str) or Path(train).suffix.lower() != ".txt":
        raise ValueError("Training source must be the weighted .txt image manifest")
    manifest = Path(train).expanduser()
    if not manifest.is_absolute():
        base = Path(data.get("path", dataset_yaml.parent))
        if not base.is_absolute():
            base = (dataset_yaml.parent / base).resolve()
        manifest = (base / manifest).resolve()
    if not manifest.is_file():
        raise FileNotFoundError(f"Weighted training manifest not found: {manifest}")
    entries = [row.strip() for row in manifest.read_text(encoding="utf-8").splitlines() if row.strip()]
    if not entries:
        raise ValueError("Weighted training manifest is empty")
    LOGGER.info("Weighted training manifest contains %d samples", len(entries))
    return {"train_manifest": str(manifest), "training_sample_entries": len(entries),
            "class_names": expected}


def collect_environment() -> dict[str, Any]:
    import torch
    import ultralytics

    return {
        "python_version": platform.python_version(),
        "pytorch_version": torch.__version__,
        "ultralytics_version": ultralytics.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


def train(data: Path, output_dir: Path, device: str, run_name: str,
          config: TrainingConfig, dry_run: bool = False) -> Path:
    """Train with Algorithm 1 manifest + Algorithm 2 on-the-fly augmentation."""
    config.validate()
    data = data.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    audit = validate_dataset_yaml(data)
    augment = detection_augmentation_kwargs()

    # Match the experiment without embedding machine-specific device paths.
    args: dict[str, Any] = {
        **asdict(config),
        **augment,
        "data": str(data),
        "task": "detect",
        "device": device,
        "project": str(output_dir),
        "name": run_name,
        "exist_ok": False,
        "save": True,
    }
    if dry_run:
        print(json.dumps({"training_arguments": args, "dataset_audit": audit}, indent=2))
        return output_dir / run_name

    # Keep experiment metadata separate from protected source images and labels.
    output_dir.mkdir(parents=True, exist_ok=True)
    run_dir = output_dir / run_name
    if run_dir.exists():
        raise FileExistsError(f"Output run already exists: {run_dir}. Choose another --run-name")

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("Ultralytics is required: pip install ultralytics") from exc

    environment = collect_environment()
    LOGGER.info("Starting CariesLoc-HB training: %s", run_name)
    model = YOLO(config.model)
    model.train(**args)

    if not run_dir.is_dir():
        raise RuntimeError("Training completed, but the requested run directory was not created")
    metadata = {
        "method": "CariesLoc-HB (YOLO11m)",
        "stage": "training",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "configuration": {key: value for key, value in args.items()
                          if key not in ("data", "project")},
        "augmentation": augment,
        "input_manifest_samples": audit["training_sample_entries"],
        "class_names": audit["class_names"],
        "environment": environment,
        "note": "No test-set or external validation results are generated by this module.",
    }
    (run_dir / "cariesloc_hb_training_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    LOGGER.info("Training finished; outputs available in %s", run_dir)
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", required=True, type=Path,
                        help="sampling_data.yaml produced by the separate sampling stage")
    parser.add_argument("--output-dir", required=True, type=Path,
                        help="Directory for model checkpoints and run logs")
    parser.add_argument("--run-name", default="cariesloc_hb_yolo11m",
                        help="Name of the experiment subdirectory")
    parser.add_argument("--device", default="0",
                        help="Ultralytics device, e.g., 0, 0,1, or cpu")
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate input and print settings without training")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    train(args.data, args.output_dir, args.device, args.run_name,
          TrainingConfig(), dry_run=args.dry_run)


if __name__ == "__main__":
    main()
