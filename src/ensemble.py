"""Weighted Boxes Fusion of cached prediction files (predictions/<split>_<tag>.json written by src/infer.py).

Every model is run once with its own best inference setting (resolution, tiles, ...), and the ensemble is
built offline, so fusion weights / IoU can be tuned on validation without re-running the detectors.

    python src/ensemble.py val  --tags rtdetr_q500_tiles2 final1280_tiles2 tilesm_last_t2 --weights 1 1 2 --tag ens3
    python src/ensemble.py test --tags ... --weights ... --tag ens3      (same tags, test_ prefix)
"""
import argparse
import json

import numpy as np
import pandas as pd
from ensemble_boxes import weighted_boxes_fusion

from common import DATA, PRED, PREP, RESULTS
from coco_eval import evaluate
from submission import to_submission
from pathlib import Path

MAX_DET = 500


def image_sizes(split: str) -> dict:
    if split == "test":
        df = pd.read_csv(DATA / "test_images.csv")
        return {r.image_id: (r.width, r.height) for r in df.itertuples()}
    gt = json.load(open(PREP / "val_s5" / "coco_gt.json"))
    return {Path(im["file_name"]).stem: (im["width"], im["height"]) for im in gt["images"]}


def fuse(preds: list[dict], sizes: dict, weights, iou: float, skip: float, conf_type: str) -> dict:
    out = {}
    for stem, (w, h) in sizes.items():
        norm = np.array([w, h, w, h], dtype=np.float32)
        boxes, scores, labels = [], [], []
        for p in preds:
            d = np.asarray(p.get(stem, []), dtype=np.float32).reshape(-1, 6)   # cls, score, x, y, w, h
            xyxy = np.c_[d[:, 2:4], d[:, 2:4] + d[:, 4:6]] / norm
            boxes.append(np.clip(xyxy, 0, 1)); scores.append(d[:, 1]); labels.append(d[:, 0])
        b, s, c = weighted_boxes_fusion(boxes, scores, labels, weights=weights, iou_thr=iou,
                                        skip_box_thr=skip, conf_type=conf_type)
        order = np.argsort(-s)[:MAX_DET]
        b = b[order] * norm
        out[stem] = [[int(ci), float(si), float(x1), float(y1), float(x2 - x1), float(y2 - y1)]
                     for (x1, y1, x2, y2), si, ci in zip(b, s[order], c[order])]
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("split", choices=["val", "test"])
    ap.add_argument("--tags", nargs="+", required=True)
    ap.add_argument("--weights", nargs="+", type=float, default=None)
    ap.add_argument("--iou", type=float, default=0.6)
    ap.add_argument("--skip", type=float, default=0.001)
    ap.add_argument("--conf-type", default="avg")
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()
    preds = [json.load(open(PRED / f"{a.split}_{t}.json")) for t in a.tags]
    out = fuse(preds, image_sizes(a.split), a.weights, a.iou, a.skip, a.conf_type)
    json.dump(out, open(PRED / f"{a.split}_{a.tag}.json", "w"))
    if a.split == "test":
        to_submission(out, Path("submissions") / f"{a.tag}.csv")
    else:
        m = evaluate(out, PREP / "val_s5" / "coco_gt.json")
        m.update(run=a.tag, members=a.tags, fusion_weights=a.weights, wbf_iou=a.iou, conf_type=a.conf_type)
        with open(RESULTS / "val_metrics.jsonl", "a") as f:
            f.write(json.dumps(m) + "\n")
        print(f"{a.tag:24s} mAP {m['mAP']:.4f} AP50 {m['AP50']:.4f} APs {m['AP_small']:.4f} APl {m['AP_large']:.4f}")
