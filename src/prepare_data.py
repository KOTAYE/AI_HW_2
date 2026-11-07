"""Build the fixed training / validation subsets shared by all detectors.

* Frames are subsampled per sequence with a fixed step (neighbouring frames are near-duplicates).
* Training images: ignored regions (crowds, "others", ...) are filled with grey so the model never
  sees them as background; GT boxes that are mostly (>50%) inside an ignored region are dropped.
* Validation images are NOT masked (test images are not masked either); ignored regions are kept as
  iscrowd=1 in the COCO ground truth so the evaluator skips detections there.
* Images are downscaled so that the longer side is at most `--max-side` px (labels are normalised,
  so they do not change). Validation GT stays in ORIGINAL pixel coordinates.

Output layout (Ultralytics-compatible):
    data/prepared/<name>/images/*.jpg, labels/*.txt
    data/prepared/<name>/coco_gt.json   (validation subsets only, original coordinates)
    data/prepared/<name>.yaml           (Ultralytics dataset config: train subset + val subset)

Usage:
    python src/prepare_data.py --split train --step 10
    python src/prepare_data.py --split val --step 5
    python src/prepare_data.py --yaml train_s10 val_s5
"""
import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

from common import CLASSES, DATA, PREP, flat_name, read_yolo_boxes

FILL = (114, 114, 114)  # same grey Ultralytics uses for letterbox padding


def select_frames(split: str, step: int) -> list[dict]:
    """Return COCO image entries of every `step`-th frame of each sequence (frame 1, 1+step, ...)."""
    coco = json.load(open(DATA / split / "annotations_coco.json"))
    return [im for im in coco["images"] if (im["frame_id"] - 1) % step == 0]


def mask_fraction(box: np.ndarray, mask: np.ndarray) -> float:
    """Fraction of a pixel box [x1, y1, x2, y2] covered by the ignore mask."""
    x1, y1, x2, y2 = box.astype(int)
    patch = mask[max(y1, 0):max(y2, 0), max(x1, 0):max(x2, 0)]
    return float(patch.mean()) if patch.size else 0.0


def process_frame(args) -> tuple[int, int]:
    im, split, out_dir, do_mask, max_side = args
    video, frame = im["video"], im["frame_id"]
    src = DATA / split / "images" / video / f"{frame:07d}.jpg"
    stem = flat_name(video, frame)
    img = cv2.imread(str(src))
    h, w = img.shape[:2]

    labels = read_yolo_boxes(DATA / split / "labels" / video / f"{frame:07d}.txt")
    dropped = 0
    if do_mask:
        ign = read_yolo_boxes(DATA / split / "ignore" / video / f"{frame:07d}.txt", has_class=False)
        if len(ign):
            mask = np.zeros((h, w), dtype=np.uint8)
            for cx, cy, bw, bh in ign:
                x1, y1 = int((cx - bw / 2) * w), int((cy - bh / 2) * h)
                x2, y2 = int(np.ceil((cx + bw / 2) * w)), int(np.ceil((cy + bh / 2) * h))
                mask[max(y1, 0):y2, max(x1, 0):x2] = 1
            keep = []
            for c, cx, cy, bw, bh in labels:
                box = np.array([(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h])
                keep.append(mask_fraction(box, mask) <= 0.5)
            keep = np.array(keep, dtype=bool)
            dropped = int((~keep).sum())
            labels = labels[keep] if len(labels) else labels
            # paint ignored pixels, but restore pixels of kept GT boxes so overlapping objects stay visible
            painted = img.copy()
            painted[mask.astype(bool)] = FILL
            for c, cx, cy, bw, bh in labels:
                x1, y1 = max(int((cx - bw / 2) * w), 0), max(int((cy - bh / 2) * h), 0)
                x2, y2 = int(np.ceil((cx + bw / 2) * w)), int(np.ceil((cy + bh / 2) * h))
                painted[y1:y2, x1:x2] = img[y1:y2, x1:x2]
            img = painted

    scale = max_side / max(h, w)
    if scale < 1:
        img = cv2.resize(img, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(out_dir / "images" / f"{stem}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    with open(out_dir / "labels" / f"{stem}.txt", "w") as f:
        for c, cx, cy, bw, bh in labels:
            f.write(f"{int(c)} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
    return len(labels), dropped


def write_coco_subset(split: str, frames: list[dict], out_path: Path) -> None:
    """COCO GT for the selected frames, original pixel coords, ignored regions kept as iscrowd=1."""
    coco = json.load(open(DATA / split / "annotations_coco.json"))
    ids = {im["id"] for im in frames}
    images = [dict(im, file_name=flat_name(im["video"], im["frame_id"]) + ".jpg") for im in frames]
    anns = [a for a in coco["annotations"] if a["image_id"] in ids]
    json.dump({"images": images, "annotations": anns, "categories": coco["categories"]}, open(out_path, "w"))


def build(split: str, step: int, mask: bool, max_side: int, workers: int) -> None:
    name = f"{split}_s{step}" + ("_nomask" if split == "train" and not mask else "")
    out = PREP / name
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "labels").mkdir(parents=True, exist_ok=True)
    frames = select_frames(split, step)
    jobs = [(im, split, out, mask, max_side) for im in frames]
    with ProcessPoolExecutor(workers) as ex:
        stats = list(ex.map(process_frame, jobs, chunksize=16))
    kept, dropped = map(sum, zip(*stats))
    if split == "val":
        write_coco_subset(split, frames, out / "coco_gt.json")
    print(f"{name}: {len(frames)} frames, {kept} boxes kept, {dropped} dropped (inside ignore)")


def write_yaml(train_name: str, val_name: str) -> Path:
    path = PREP / f"{train_name}__{val_name}.yaml"
    names = "\n".join(f"  {i}: {n}" for i, n in enumerate(CLASSES))
    path.write_text(f"path: {PREP.as_posix()}\ntrain: {train_name}/images\nval: {val_name}/images\nnames:\n{names}\n")
    print("wrote", path)
    return path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "val"])
    ap.add_argument("--step", type=int, default=10)
    ap.add_argument("--no-mask", action="store_true", help="do not paint ignored regions")
    ap.add_argument("--max-side", type=int, default=1920)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--yaml", nargs=2, metavar=("TRAIN", "VAL"), help="only write a dataset yaml")
    a = ap.parse_args()
    if a.yaml:
        write_yaml(*a.yaml)
    else:
        # validation subsets are never masked (they must look like test images)
        build(a.split, a.step, mask=(a.split == "train" and not a.no_mask), max_side=a.max_side, workers=a.workers)
