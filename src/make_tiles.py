"""Cut a prepared YOLO split into overlapping n x n tiles for tile-based training.

VisDrone objects are tiny (EDA: median ~24 px after resizing to 1024), so instead of shrinking the whole
frame we train on crops that keep the native resolution. The same tile grid is used at test time
(src/infer.py --tiles), so the detector sees objects at the scale it was trained on.

Boxes are clipped to the tile; a box is kept if at least `--min-vis` of its area stays inside
(otherwise the cut fragment is dropped, as in SAHI slicing).

    python src/make_tiles.py --src train_s5 --n 2 --overlap 0.2
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import partial

import cv2
import numpy as np

from common import PREP, read_yolo_boxes
from infer import tiles_of
from prepare_data import write_yaml


def cut(img_path, dst, n, overlap, min_vis):
    img = cv2.imread(str(img_path))
    h, w = img.shape[:2]
    lab = read_yolo_boxes(img_path.parent.parent / "labels" / f"{img_path.stem}.txt")
    cls = lab[:, 0]
    xyxy = np.c_[(lab[:, 1] - lab[:, 3] / 2) * w, (lab[:, 2] - lab[:, 4] / 2) * h,
                 (lab[:, 1] + lab[:, 3] / 2) * w, (lab[:, 2] + lab[:, 4] / 2) * h]
    area = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
    kept = 0
    for i, (x1, y1, x2, y2) in enumerate(tiles_of(h, w, n, overlap)):
        b = xyxy.copy()
        b[:, [0, 2]] = b[:, [0, 2]].clip(x1, x2) - x1
        b[:, [1, 3]] = b[:, [1, 3]].clip(y1, y2) - y1
        bw, bh = b[:, 2] - b[:, 0], b[:, 3] - b[:, 1]
        m = (bw * bh >= min_vis * area) & (bw >= 2) & (bh >= 2)
        tw, th = x2 - x1, y2 - y1
        rows = [f"{int(c)} {(bx1 + bx2) / 2 / tw:.6f} {(by1 + by2) / 2 / th:.6f} {(bx2 - bx1) / tw:.6f} {(by2 - by1) / th:.6f}"
                for c, (bx1, by1, bx2, by2) in zip(cls[m], b[m])]
        stem = f"{img_path.stem}__t{i}"
        cv2.imwrite(str(dst / "images" / f"{stem}.jpg"), img[y1:y2, x1:x2], [cv2.IMWRITE_JPEG_QUALITY, 95])
        (dst / "labels" / f"{stem}.txt").write_text("\n".join(rows))
        kept += int(m.sum())
    return kept


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="train_s5")
    ap.add_argument("--n", type=int, default=2)
    ap.add_argument("--overlap", type=float, default=0.2)
    ap.add_argument("--min-vis", type=float, default=0.5)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--val", default="val_s5")
    a = ap.parse_args()
    name = f"{a.src}_t{a.n}"
    dst = PREP / name
    for sub in ["images", "labels"]:
        (dst / sub).mkdir(parents=True, exist_ok=True)
    imgs = sorted((PREP / a.src / "images").glob("*.jpg"))
    with ProcessPoolExecutor(a.workers) as ex:
        boxes = sum(ex.map(partial(cut, dst=dst, n=a.n, overlap=a.overlap, min_vis=a.min_vis), imgs, chunksize=16))
    print(f"{name}: {len(imgs) * a.n * a.n} tiles, {boxes} boxes")
    write_yaml(name, a.val)
