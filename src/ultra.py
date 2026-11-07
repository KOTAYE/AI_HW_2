"""Train / predict / evaluate Ultralytics models (YOLO = anchor-free, RT-DETR = query-based).

All models share the fixed protocol: train on data/prepared/train_s10, validate on val_s5 with
coco_eval.py (ignored regions excluded, original-resolution GT).

Examples:
    python src/ultra.py train --model yolo11s.pt --name yolo11s_1024 --imgsz 1024 --epochs 30
    python src/ultra.py eval  --name yolo11s_1024 --imgsz 1024
    python src/ultra.py test  --name yolo11s_1024 --imgsz 1024
Extra Ultralytics arguments can be passed as key=value after `--set`, e.g. `--set box=10 cls=0.3`.
"""
import argparse
import json
import time
from pathlib import Path

import pandas as pd
import torch

from common import DATA, PRED, PREP, RESULTS, RUNS, SEED, parse_flat_name, seed_everything
from coco_eval import evaluate
from submission import to_submission


def parse_set(items: list[str]) -> dict:
    out = {}
    for it in items or []:
        k, v = it.split("=", 1)
        try:
            v = json.loads(v)
        except json.JSONDecodeError:
            pass
        out[k] = v
    return out


def load_model(weights: str):
    from ultralytics import RTDETR, YOLO
    return RTDETR(weights) if "rtdetr" in Path(weights).name.lower() else YOLO(weights)


def train(a) -> None:
    seed_everything()
    model = load_model(a.model)
    cfg = dict(data=str(PREP / f"{a.train}__{a.val}.yaml"), imgsz=a.imgsz, epochs=a.epochs, batch=a.batch,
               project=str(RUNS), name=a.name, exist_ok=True, seed=SEED, deterministic=False,
               workers=6, cos_lr=True, close_mosaic=max(1, a.epochs // 10), max_det=500,
               patience=0, plots=True, amp=True, cache=a.cache)
    cfg.update(parse_set(a.set))
    if a.cfg_yaml:  # e.g. custom architecture (P2 head) built from yaml + pretrained weights
        model = load_model(a.cfg_yaml).load(a.model)
    model.train(**cfg)


def val_items() -> list[tuple[str, Path]]:
    """(flat stem, ORIGINAL image path) of the fixed validation subset."""
    gt = json.load(open(PREP / "val_s5" / "coco_gt.json"))
    items = []
    for im in gt["images"]:
        stem = Path(im["file_name"]).stem
        video, frame = parse_flat_name(stem)
        items.append((stem, DATA / "val" / "images" / video / f"{frame:07d}.jpg"))
    return items


def test_items() -> list[tuple[str, Path]]:
    df = pd.read_csv(DATA / "test_images.csv")
    return [(r.image_id, DATA / r.file_name) for r in df.itertuples()]


@torch.no_grad()
def predict(weights: Path, items, imgsz: int, conf: float = 0.001, iou: float = 0.6,
            augment: bool = False, batch: int = 8) -> tuple[dict, float]:
    """Run inference; returns ({stem: [[cls, score, x, y, w, h], ...]}, ms per image)."""
    model = load_model(str(weights))
    preds, t_total = {}, 0.0
    for i in range(0, len(items), batch):
        chunk = items[i:i + batch]
        t0 = time.perf_counter()
        results = model.predict([str(p) for _, p in chunk], imgsz=imgsz, conf=conf, iou=iou, max_det=500,
                                augment=augment, verbose=False, half=True)
        t_total += time.perf_counter() - t0
        for (stem, _), r in zip(chunk, results):
            b = r.boxes
            xyxy = b.xyxy.cpu().numpy()
            dets = [[int(c), float(s), float(x1), float(y1), float(x2 - x1), float(y2 - y1)]
                    for (x1, y1, x2, y2), s, c in zip(xyxy, b.conf.cpu().numpy(), b.cls.cpu().numpy())]
            preds[stem] = dets
    return preds, 1000 * t_total / len(items)


def weights_of(name: str) -> Path:
    return RUNS / name / "weights" / "best.pt"


def evaluate_run(a) -> dict:
    tag = a.name + (f"_{a.tag}" if a.tag else "")
    preds, ms = predict(weights_of(a.name), val_items(), a.imgsz, conf=a.conf, iou=a.iou, augment=a.augment)
    PRED.mkdir(exist_ok=True)
    json.dump(preds, open(PRED / f"val_{tag}.json", "w"))
    m = evaluate(preds, PREP / "val_s5" / "coco_gt.json")
    m.update(run=tag, imgsz=a.imgsz, ms_per_img=round(ms, 1))
    RESULTS.mkdir(exist_ok=True)
    with open(RESULTS / "val_metrics.jsonl", "a") as f:
        f.write(json.dumps(m) + "\n")
    print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in m.items()}, indent=1))
    return m


def test_run(a) -> None:
    tag = a.name + (f"_{a.tag}" if a.tag else "")
    preds, _ = predict(weights_of(a.name), test_items(), a.imgsz, conf=a.conf, iou=a.iou, augment=a.augment)
    json.dump(preds, open(PRED / f"test_{tag}.json", "w"))
    to_submission(preds, Path("submissions") / f"{tag}.csv")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["train", "eval", "test"])
    ap.add_argument("--model", default="yolo11s.pt", help="pretrained weights")
    ap.add_argument("--cfg-yaml", default=None, help="optional architecture yaml (weights loaded from --model)")
    ap.add_argument("--name", required=True)
    ap.add_argument("--train", default="train_s10")
    ap.add_argument("--val", default="val_s5")
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--cache", default=False)
    ap.add_argument("--conf", type=float, default=0.001)
    ap.add_argument("--iou", type=float, default=0.6)
    ap.add_argument("--augment", action="store_true", help="Ultralytics test-time augmentation")
    ap.add_argument("--tag", default="", help="suffix for prediction files (e.g. inference variants)")
    ap.add_argument("--set", nargs="*", help="extra Ultralytics train args key=value")
    a = ap.parse_args()
    {"train": train, "eval": evaluate_run, "test": test_run}[a.mode](a)
