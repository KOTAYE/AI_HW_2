"""COCO evaluation of cached predictions on a validation subset, identical for every detector.

Predictions format (shared by all models), saved as JSON:
    {"<flat stem>": [[cls0, score, x, y, w, h], ...], ...}
with 0-based class ids and boxes in ORIGINAL image pixels, (x, y) = top-left corner — exactly the
Kaggle submission format, so the same file can be turned into a submission for test images.

Ignored regions are iscrowd=1 in the GT, so detections inside them are not counted as false positives.
maxDets is raised to 500 (the Kaggle cap per image) because frames contain ~40 objects on average.
"""
import argparse
import io
import json
import contextlib
from pathlib import Path

import numpy as np

try:  # faster drop-in replacement for pycocotools
    from faster_coco_eval import COCO, COCOeval_faster as COCOeval
except ImportError:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

from common import CLASSES, PREP, flat_name

STAT_NAMES = ["mAP", "AP50", "AP75", "AP_small", "AP_medium", "AP_large",
              "AR1", "AR10", "AR500", "AR_small", "AR_medium", "AR_large"]


def to_coco_results(preds: dict, gt: COCO) -> list[dict]:
    stem_to_id = {Path(im["file_name"]).stem: im["id"] for im in gt.dataset["images"]}
    res = []
    for stem, dets in preds.items():
        if stem not in stem_to_id:
            continue
        for c, s, x, y, w, h in dets:
            res.append({"image_id": stem_to_id[stem], "category_id": int(c) + 1,
                        "bbox": [float(x), float(y), float(w), float(h)], "score": float(s)})
    return res


def evaluate(preds: dict, gt_path: Path, per_class: bool = True, verbose: bool = False) -> dict:
    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO(str(gt_path))
    res = to_coco_results(preds, gt)
    if not res:
        return {k: 0.0 for k in STAT_NAMES}
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        dt = gt.loadRes(res)
        ev = COCOeval(gt, dt, "bbox")
        ev.params.maxDets = [1, 10, 500]
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
    if verbose:
        print(buf.getvalue())
    out = dict(zip(STAT_NAMES, map(float, ev.stats[:12])))
    if per_class:
        # precision: [T, R, K, A, M] -> AP@[.5:.95] per class, area=all, maxDets=500
        prec = ev.eval["precision"][:, :, :, 0, -1]
        for k, name in enumerate(CLASSES):
            p = prec[:, :, k]
            out[f"AP_{name}"] = float(p[p > -1].mean()) if (p > -1).any() else float("nan")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("preds", type=Path, help="predictions json")
    ap.add_argument("--gt", type=Path, default=PREP / "val_s5" / "coco_gt.json")
    a = ap.parse_args()
    m = evaluate(json.load(open(a.preds)), a.gt, verbose=True)
    print(json.dumps({k: round(v, 4) for k, v in m.items()}, indent=1))
