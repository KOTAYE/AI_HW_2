"""Convert cached test predictions (same JSON format as coco_eval.py) into a Kaggle submission CSV."""
import argparse
import json
from pathlib import Path

import pandas as pd

from common import DATA

MAX_DETS = 500  # Kaggle scores at most 500 detections per image


def test_stem(row) -> str:
    return row.image_id  # e.g. clip_001_0000001


def to_submission(preds: dict, out_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(DATA / "test_images.csv")
    rows = []
    n_missing = 0
    for r in df.itertuples():
        dets = preds.get(r.image_id)
        if dets is None:
            n_missing += 1
            dets = []
        dets = sorted(dets, key=lambda d: -d[1])[:MAX_DETS]
        parts = []
        for c, s, x, y, w, h in dets:
            # clip to image and drop degenerate boxes
            x1, y1 = max(0.0, x), max(0.0, y)
            x2, y2 = min(float(r.width), x + w), min(float(r.height), y + h)
            if x2 - x1 < 1 or y2 - y1 < 1:
                continue
            parts.append(f"{int(c)} {s:.4f} {x1:.1f} {y1:.1f} {x2 - x1:.1f} {y2 - y1:.1f}")
        rows.append((r.image_id, " ".join(parts)))
    sub = pd.DataFrame(rows, columns=["image_id", "PredictionString"])
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(out_csv, index=False)
    print(f"wrote {out_csv}: {len(sub)} rows, {n_missing} images without predictions entry, "
          f"{sum(len(p.split()) // 6 for p in sub.PredictionString)} boxes")
    return sub


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("preds", type=Path)
    ap.add_argument("out", type=Path)
    a = ap.parse_args()
    to_submission(json.load(open(a.preds)), a.out)
