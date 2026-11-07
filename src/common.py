"""Shared paths, constants and helpers used by every model in this project."""
from pathlib import Path
import random

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"              # raw Kaggle data: data/train, data/val, data/test, data/test_images.csv
PREP = DATA / "prepared"          # outputs of prepare_data.py
RUNS = ROOT / "runs"              # training outputs (weights, logs)
PRED = ROOT / "predictions"       # cached validation / test predictions
RESULTS = ROOT / "results"        # metric tables (csv/json) used in the report

CLASSES = ["person", "bicycle", "car", "van", "truck", "tricycle", "bus", "motor"]
NUM_CLASSES = len(CLASSES)
SEED = 42


def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def flat_name(video: str, frame: int) -> str:
    """Unique flat file stem for a frame, e.g. 'uav0000086_00000_v__0000001'."""
    return f"{video}__{frame:07d}"


def parse_flat_name(stem: str) -> tuple[str, int]:
    video, frame = stem.split("__")
    return video, int(frame)


def read_yolo_boxes(path: Path, has_class: bool = True) -> np.ndarray:
    """Read a YOLO txt file -> array (N, 5) [cls, cx, cy, w, h] or (N, 4) [cx, cy, w, h] (normalised)."""
    ncol = 5 if has_class else 4
    if not path.exists() or path.stat().st_size == 0:
        return np.zeros((0, ncol), dtype=np.float32)
    arr = np.loadtxt(path, dtype=np.float32, ndmin=2)
    return arr.reshape(-1, ncol)
