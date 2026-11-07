"""Inference strategies on top of trained Ultralytics models (used in the "Your pipeline" part).

* resolution: run the detector at a different test size than training
* tta:        Ultralytics test-time augmentation (flips + scales)
* tiling:     SAHI-style overlapping tiles + the full frame, merged per class (SAHI, Akyon et al. 2022)
* ensemble:   several models merged with Weighted Boxes Fusion (Solovyev et al. 2019, `ensemble_boxes`)
* temporal:   score smoothing with matched detections in neighbouring frames (Seq-NMS / T-CNN-style
              rescoring, Han et al. 2016; Kang et al. 2016) - consistent objects keep their score, flickering
              false positives are suppressed. Uses consecutive frames, so it runs on whole clips.

Examples:
    python src/infer.py val  --weights runs/yolo11s_1024/weights/best.pt --imgsz 1280 --tag r1280
    python src/infer.py val  --weights runs/a/weights/best.pt --tiles 2 --tag tiles2
    python src/infer.py val  --weights runs/a/weights/best.pt --temporal 2 --tag temporal
    python src/infer.py test --weights runs/a/weights/best.pt runs/b/weights/best.pt --tag ens
"""
import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from ensemble_boxes import weighted_boxes_fusion
from torchvision.ops import batched_nms, box_iou

from common import DATA, PRED, PREP, RESULTS, parse_flat_name
from coco_eval import evaluate
from submission import to_submission
from ultra import load_model

MAX_DET = 500


# ----------------------------------------------------------------------------- single model
def _to_array(r) -> np.ndarray:
    b = r.boxes
    return np.concatenate([b.xyxy.cpu().numpy(), b.conf.cpu().numpy()[:, None], b.cls.cpu().numpy()[:, None]], 1)


def tiles_of(h: int, w: int, n: int, overlap: float) -> list[tuple[int, int, int, int]]:
    """n x n overlapping tiles (x1, y1, x2, y2)."""
    tw, th = int(w / (n - (n - 1) * overlap)), int(h / (n - (n - 1) * overlap))
    xs = np.linspace(0, w - tw, n).astype(int)
    ys = np.linspace(0, h - th, n).astype(int)
    return [(x, y, x + tw, y + th) for y in ys for x in xs]


def nms_merge(d: np.ndarray, iou: float) -> np.ndarray:
    if len(d) == 0:
        return d
    t = torch.from_numpy(d).float()
    keep = batched_nms(t[:, :4], t[:, 4], t[:, 5].long(), iou)[:MAX_DET]
    return d[keep.numpy()]


def wbf_merge(dets: list[np.ndarray], w: int, h: int, iou: float, weights=None) -> np.ndarray:
    norm = np.array([w, h, w, h], dtype=np.float32)
    boxes = [np.clip(d[:, :4] / norm, 0, 1) for d in dets]
    b, s, c = weighted_boxes_fusion(boxes, [d[:, 4] for d in dets], [d[:, 5] for d in dets],
                                    weights=weights, iou_thr=iou, skip_box_thr=0.001, conf_type="avg")
    out = np.concatenate([b * norm, s[:, None], c[:, None]], 1)
    return out[np.argsort(-out[:, 4])][:MAX_DET]


def detect(model, img: np.ndarray, a) -> np.ndarray:
    """Detections (N, 6) [x1, y1, x2, y2, score, cls] in original pixels for one BGR image."""
    h, w = img.shape[:2]
    kw = dict(imgsz=a.imgsz, conf=a.conf, iou=a.iou, max_det=MAX_DET, verbose=False, half=True)
    full = _to_array(model.predict(img, augment=a.tta, **kw)[0])
    if a.tiles <= 1:
        return full
    boxes = tiles_of(h, w, a.tiles, a.overlap)
    crops = [img[y1:y2, x1:x2] for x1, y1, x2, y2 in boxes]
    parts = [full]
    for (x1, y1, x2, y2), r in zip(boxes, model.predict(crops, **kw)):
        d = _to_array(r)
        # drop boxes cut by an inner tile border (they are covered by the neighbouring tile / full frame)
        m = np.ones(len(d), bool)
        if x1 > 0: m &= d[:, 0] > 2
        if y1 > 0: m &= d[:, 1] > 2
        if x2 < w: m &= d[:, 2] < (x2 - x1) - 2
        if y2 < h: m &= d[:, 3] < (y2 - y1) - 2
        d = d[m]
        d[:, [0, 2]] += x1
        d[:, [1, 3]] += y1
        parts.append(d)
    if a.merge == "wbf":
        return wbf_merge(parts, w, h, a.merge_iou)
    return nms_merge(np.concatenate(parts), a.merge_iou)


# ----------------------------------------------------------------------------- temporal rescoring
def temporal_rescore(preds: dict, frame_of: dict, window: int, iou_thr: float = 0.5) -> dict:
    """score_t <- mean over frames t-k..t+k of the best same-class matching score (0 if no match).

    frame_of: stem -> (clip, frame). Frames missing at clip borders are skipped (not counted as zero).
    """
    by_clip = defaultdict(dict)
    for stem, (clip, f) in frame_of.items():
        if stem in preds:
            by_clip[clip][f] = stem
    out = {}
    for clip, frames in by_clip.items():
        arr = {f: np.asarray(preds[s], dtype=np.float32).reshape(-1, 6) for f, s in frames.items()}
        for f, stem in frames.items():
            d = arr[f]
            if len(d) == 0:
                out[stem] = []
                continue
            xyxy = torch.from_numpy(np.c_[d[:, 2:4], d[:, 2:4] + d[:, 4:6]])
            acc, cnt = d[:, 1].copy(), np.ones(len(d), np.float32)
            for k in range(-window, window + 1):
                if k == 0 or (f + k) not in arr:
                    continue
                n = arr[f + k]
                cnt += 1
                if len(n) == 0:
                    continue
                nb = torch.from_numpy(np.c_[n[:, 2:4], n[:, 2:4] + n[:, 4:6]])
                iou = box_iou(xyxy, nb).numpy()
                iou[d[:, 0][:, None] != n[:, 0][None, :]] = 0          # same class only
                ok = iou >= iou_thr
                best = np.where(ok, n[:, 1][None, :], 0).max(1)
                acc += best
            new = d.copy()
            new[:, 1] = acc / cnt
            out[stem] = new.tolist()
    return out


# ----------------------------------------------------------------------------- driver
def items_for(split: str, temporal: int) -> tuple[list, dict]:
    """[(stem, path)] to run on, and stem -> (clip, frame) for temporal rescoring."""
    if split == "test":
        df = pd.read_csv(DATA / "test_images.csv")
        items = [(r.image_id, DATA / r.file_name) for r in df.itertuples()]
        frame_of = {r.image_id: (r.clip, r.frame) for r in df.itertuples()}
        return items, frame_of
    gt = json.load(open(PREP / "val_s5" / "coco_gt.json"))
    stems = [Path(im["file_name"]).stem for im in gt["images"]]
    need = set()
    for s in stems:  # evaluation frames + their temporal neighbours
        v, f = parse_flat_name(s)
        for k in range(-temporal, temporal + 1):
            need.add((v, f + k))
    items, frame_of = [], {}
    for v, f in sorted(need):
        p = DATA / "val" / "images" / v / f"{f:07d}.jpg"
        if p.exists():
            stem = f"{v}__{f:07d}"
            items.append((stem, p))
            frame_of[stem] = (v, f)
    return items, frame_of


def run(a) -> None:
    items, frame_of = items_for(a.split, a.temporal)
    models = [load_model(w) for w in a.weights]
    preds, t_total = {}, 0.0
    for i, (stem, path) in enumerate(items):
        img = cv2.imread(str(path))
        t0 = time.perf_counter()
        dets = [detect(m, img, a) for m in models]
        d = dets[0] if len(dets) == 1 else wbf_merge(dets, img.shape[1], img.shape[0], a.merge_iou)
        t_total += time.perf_counter() - t0
        preds[stem] = [[int(c), float(s), float(x1), float(y1), float(x2 - x1), float(y2 - y1)]
                       for x1, y1, x2, y2, s, c in d]
        if i % 500 == 0:
            print(f"{i}/{len(items)}", flush=True)
    if a.temporal:
        preds = temporal_rescore(preds, frame_of, a.temporal, a.temporal_iou)
    PRED.mkdir(exist_ok=True)
    json.dump(preds, open(PRED / f"{a.split}_{a.tag}.json", "w"))
    if a.split == "test":
        to_submission(preds, Path("submissions") / f"{a.tag}.csv")
        return
    m = evaluate(preds, PREP / "val_s5" / "coco_gt.json")
    m.update(run=a.tag, imgsz=a.imgsz, ms_per_img=round(1000 * t_total / len(items), 1),
             weights=[str(w) for w in a.weights])
    RESULTS.mkdir(exist_ok=True)
    with open(RESULTS / "val_metrics.jsonl", "a") as f:
        f.write(json.dumps(m) + "\n")
    print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in m.items()}, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("split", choices=["val", "test"])
    ap.add_argument("--weights", nargs="+", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--conf", type=float, default=0.001)
    ap.add_argument("--iou", type=float, default=0.6, help="NMS IoU inside the detector")
    ap.add_argument("--tta", action="store_true")
    ap.add_argument("--tiles", type=int, default=1, help="n for an n x n tile grid (1 = no tiling)")
    ap.add_argument("--overlap", type=float, default=0.2)
    ap.add_argument("--merge", choices=["nms", "wbf"], default="nms", help="how tiles are merged")
    ap.add_argument("--merge-iou", type=float, default=0.6)
    ap.add_argument("--temporal", type=int, default=0, help="temporal window k (0 = off)")
    ap.add_argument("--temporal-iou", type=float, default=0.5)
    a = ap.parse_args()
    run(a)
