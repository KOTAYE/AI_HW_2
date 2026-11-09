"""Grid search of temporal rescoring parameters on cached raw validation predictions
(predictions/val_<tag>_raw.json, written by `src/infer.py val ... --temporal K`).

    python src/tune_temporal.py --tag A_temp3
"""
import argparse
import itertools
import json

from common import PRED, PREP, RESULTS
from coco_eval import evaluate
from infer import temporal_rescore

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--windows", nargs="+", type=int, default=[1, 2, 3])
    ap.add_argument("--ious", nargs="+", type=float, default=[0.3, 0.5])
    ap.add_argument("--alphas", nargs="+", type=float, default=[0.0, 0.25, 0.5, 1.0])
    a = ap.parse_args()
    raw = json.load(open(PRED / f"val_{a.tag}_raw.json"))
    frame_of = {k: tuple(v) for k, v in raw["frame_of"].items()}
    gt = PREP / "val_s5" / "coco_gt.json"
    stems = {im["file_name"].rsplit(".", 1)[0] for im in json.load(open(gt))["images"]}
    rows = []
    for k, iou, al in itertools.product(a.windows, a.ious, a.alphas):
        if al == 0 and (k, iou) != (a.windows[0], a.ious[0]):
            continue  # alpha 0 = no rescoring, evaluate once
        out = temporal_rescore(raw["preds"], frame_of, k, iou, al) if al else raw["preds"]
        m = evaluate({s: out[s] for s in stems if s in out}, gt)
        rows.append(dict(run=f"{a.tag}_k{k}_iou{iou}_a{al}", window=k, iou=iou, alpha=al, mAP=m["mAP"],
                         AP50=m["AP50"], AP_small=m["AP_small"]))
        print(f"k={k} iou={iou} alpha={al:4}  mAP {m['mAP']:.4f} AP50 {m['AP50']:.4f} APs {m['AP_small']:.4f}", flush=True)
    with open(RESULTS / "temporal_grid.jsonl", "a") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
