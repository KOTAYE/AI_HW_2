"""Torchvision detectors: Faster R-CNN (two-stage, anchor-based) and RetinaNet (one-stage, anchor-based).

Same protocol as ultra.py: train on data/prepared/train_s10, evaluate on val_s5 (original resolution)
with coco_eval.py. Input size is controlled by --max-size (longer side, default 1024 = YOLO imgsz).

Examples:
    python src/tv.py train --arch frcnn --name frcnn_base
    python src/tv.py train --arch frcnn --name frcnn_small_anchors --anchors 16 32 64 128 256
    python src/tv.py train --arch retina --name retina_base
    python src/tv.py eval  --arch frcnn --name frcnn_base
"""
import argparse
import json
import math
import time
from functools import partial
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader, Dataset
from torchvision.models.detection import (FasterRCNN_ResNet50_FPN_V2_Weights, RetinaNet_ResNet50_FPN_V2_Weights,
                                          fasterrcnn_resnet50_fpn_v2, retinanet_resnet50_fpn_v2)
from torchvision.models.detection import roi_heads as tv_roi_heads
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.retinanet import RetinaNetClassificationHead, RetinaNetHead
from torchvision.ops import generalized_box_iou_loss, sigmoid_focal_loss

from common import NUM_CLASSES, PRED, PREP, RESULTS, RUNS, read_yolo_boxes, seed_everything
from coco_eval import evaluate
from submission import to_submission
from ultra import test_items, val_items

DEVICE = "cuda"


# ----------------------------------------------------------------------------- data
class YoloFolder(Dataset):
    """Prepared images + YOLO labels -> (tensor CHW float [0,1], target dict with pixel xyxy, labels 1..8)."""

    def __init__(self, root: Path, train: bool = True):
        self.images = sorted((root / "images").glob("*.jpg"))
        self.labels = root / "labels"
        self.train = train

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        p = self.images[i]
        img = cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        lab = read_yolo_boxes(self.labels / f"{p.stem}.txt")
        cls, (cx, cy, bw, bh) = lab[:, 0], lab[:, 1:].T
        boxes = np.stack([(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h], 1)
        boxes = boxes.clip(0, [w, h, w, h]).astype(np.float32)
        keep = (boxes[:, 2] - boxes[:, 0] >= 1) & (boxes[:, 3] - boxes[:, 1] >= 1)
        boxes, cls = boxes[keep], cls[keep]
        if self.train and np.random.rand() < 0.5:  # horizontal flip
            img = img[:, ::-1].copy()
            boxes[:, [0, 2]] = w - boxes[:, [2, 0]]
        img = torch.from_numpy(img).permute(2, 0, 1).float() / 255
        return img, {"boxes": torch.from_numpy(boxes).reshape(-1, 4), "labels": torch.from_numpy(cls).long() + 1}


def collate(batch):
    return tuple(zip(*batch))


# ----------------------------------------------------------------------------- loss variants
def fastrcnn_loss_giou(class_logits, box_regression, labels, regression_targets, *, proposals_ref):
    """Faster R-CNN second-stage loss with GIoU instead of Smooth-L1 on decoded boxes."""
    labels = torch.cat(labels, 0)
    regression_targets = torch.cat(regression_targets, 0)
    cls_loss = torch.nn.functional.cross_entropy(class_logits, labels)
    pos = torch.where(labels > 0)[0]
    labels_pos = labels[pos]
    n, _ = class_logits.shape
    box_regression = box_regression.reshape(n, box_regression.size(-1) // 4, 4)
    props = torch.cat(proposals_ref["props"], 0)[pos]
    coder = proposals_ref["coder"]
    pred = coder.decode_single(box_regression[pos, labels_pos], props)
    tgt = coder.decode_single(regression_targets[pos], props)
    box_loss = generalized_box_iou_loss(pred, tgt, reduction="sum") / labels.numel()
    return cls_loss, box_loss


def patch_frcnn_giou(model):
    """Wrap roi_heads.select_training_samples to remember proposals, then swap the loss function."""
    rh = model.roi_heads
    ref = {"coder": rh.box_coder}
    orig = rh.select_training_samples

    def select(proposals, targets):
        out = orig(proposals, targets)
        ref["props"] = out[0]
        return out

    rh.select_training_samples = select
    tv_roi_heads.fastrcnn_loss = partial(fastrcnn_loss_giou, proposals_ref=ref)


class FocalHead(RetinaNetClassificationHead):
    """RetinaNet classification head with configurable focal-loss alpha / gamma."""

    def __init__(self, *args, alpha=0.25, gamma=2.0, **kw):
        super().__init__(*args, **kw)
        self.alpha, self.gamma = alpha, gamma

    def compute_loss(self, targets, head_outputs, matched_idxs):
        losses = []
        cls_logits = head_outputs["cls_logits"]
        for tgt, logits, midx in zip(targets, cls_logits, matched_idxs):
            fg = midx >= 0
            n_fg = fg.sum()
            gt = torch.zeros_like(logits)
            gt[fg, tgt["labels"][midx[fg]]] = 1.0
            valid = midx != self.BETWEEN_THRESHOLDS
            losses.append(sigmoid_focal_loss(logits[valid], gt[valid], alpha=self.alpha, gamma=self.gamma,
                                             reduction="sum") / max(1, n_fg))
        return sum(losses) / len(targets)


# ----------------------------------------------------------------------------- models
def build(a):
    nc = NUM_CLASSES + 1  # + background
    common = dict(min_size=int(a.max_size * 9 / 16), max_size=a.max_size)
    if a.arch == "frcnn":
        kw = dict(rpn_pre_nms_top_n_train=a.rpn_pre, rpn_post_nms_top_n_train=a.rpn_post,
                  rpn_pre_nms_top_n_test=a.rpn_pre, rpn_post_nms_top_n_test=a.rpn_post,
                  box_batch_size_per_image=a.roi_batch, box_positive_fraction=a.roi_pos,
                  box_detections_per_img=500, box_score_thresh=0.001, **common)
        model = fasterrcnn_resnet50_fpn_v2(weights=FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1, **kw)
        if a.anchors:  # one size per FPN level x len(ratios) ratios; 3 ratios keep the pretrained RPN head shape
            assert len(a.ratios) == 3, "pretrained RPN head expects 3 anchors per location"
            model.rpn.anchor_generator = AnchorGenerator(tuple((s,) for s in a.anchors),
                                                         (tuple(a.ratios),) * len(a.anchors))
        model.roi_heads.box_predictor = FastRCNNPredictor(1024, nc)
        if a.box_loss == "giou":
            patch_frcnn_giou(model)
    else:
        nc = NUM_CLASSES  # sigmoid classifier: no background class, labels 0..7
        model = retinanet_resnet50_fpn_v2(weights=RetinaNet_ResNet50_FPN_V2_Weights.COCO_V1,
                                          score_thresh=0.001, detections_per_img=500, topk_candidates=1000,
                                          **common)
        if a.anchors:
            sizes = tuple((s, int(s * 2 ** (1 / 3)), int(s * 2 ** (2 / 3))) for s in a.anchors)
            model.anchor_generator = AnchorGenerator(sizes, (tuple(a.ratios),) * len(sizes))
        na = model.anchor_generator.num_anchors_per_location()[0]
        head = RetinaNetHead(256, na, nc, norm_layer=partial(torch.nn.GroupNorm, 32))
        head.classification_head = FocalHead(256, na, nc, norm_layer=partial(torch.nn.GroupNorm, 32),
                                             alpha=a.focal_alpha, gamma=a.focal_gamma)
        head.regression_head._loss_type = a.box_loss if a.box_loss != "default" else "l1"
        # keep pretrained COCO weights of the shared conv towers and box regression
        old = model.head
        head.regression_head.load_state_dict(old.regression_head.state_dict(), strict=False) if na == 9 else None
        head.classification_head.conv.load_state_dict(old.classification_head.conv.state_dict())
        model.head = head
        model.proposal_matcher.high_threshold, model.proposal_matcher.low_threshold = a.iou_pos, a.iou_neg
    model.label_offset = 1 if a.arch == "retina" else 0  # dataset labels are 1..8
    return model


# ----------------------------------------------------------------------------- train / eval
def train(a):
    seed_everything()
    out = RUNS / a.name
    out.mkdir(parents=True, exist_ok=True)
    json.dump(vars(a), open(out / "args.json", "w"), indent=1)
    model = build(a).to(DEVICE)
    ds = YoloFolder(PREP / a.train, train=True)
    if a.limit:  # debugging only
        ds.images = ds.images[:a.limit]
    dl = DataLoader(ds, batch_size=a.batch, shuffle=True, num_workers=6, collate_fn=collate,
                    persistent_workers=True, drop_last=True)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.SGD(params, lr=a.lr, momentum=0.9, weight_decay=1e-4)
    total = a.epochs * len(dl)
    warm = min(500, total // 2)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / (total - warm))))
    scaler = torch.amp.GradScaler()
    log = []
    for ep in range(a.epochs):
        model.train()
        t0, run = time.time(), {}
        for it, (imgs, tgts) in enumerate(dl):
            imgs = [i.to(DEVICE, non_blocking=True) for i in imgs]
            tgts = [{"boxes": t["boxes"].to(DEVICE), "labels": t["labels"].to(DEVICE) - model.label_offset}
                    for t in tgts]
            with torch.autocast("cuda", dtype=torch.float16):
                losses = model(imgs, tgts)
            loss = sum(losses.values())
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, 10.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            for k, v in losses.items():
                run[k] = run.get(k, 0) + v.item()
            if it % 100 == 0:
                print(f"ep {ep} it {it}/{len(dl)} loss {loss.item():.3f} lr {opt.param_groups[0]['lr']:.4f}", flush=True)
        rec = {"epoch": ep, "time_s": round(time.time() - t0), **{k: v / len(dl) for k, v in run.items()}}
        if (ep + 1) % a.eval_every == 0 or ep + 1 == a.epochs:
            preds, _ = predict(model, val_items()[:a.limit or None])
            m = evaluate(preds, PREP / "val_s5" / "coco_gt.json", per_class=False)
            rec.update(mAP=m["mAP"], AP50=m["AP50"], AP_small=m["AP_small"])
        log.append(rec)
        print(json.dumps(rec), flush=True)
        json.dump(log, open(out / "log.json", "w"), indent=1)
        torch.save(model.state_dict(), out / "last.pth")


@torch.no_grad()
def predict(model, items, batch: int = 4) -> tuple[dict, float]:
    model.eval()
    preds, t_total = {}, 0.0
    for i in range(0, len(items), batch):
        chunk = items[i:i + batch]
        imgs = [torch.from_numpy(cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)).permute(2, 0, 1)
                .to(DEVICE).float() / 255 for _, p in chunk]
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.autocast("cuda", dtype=torch.float16):
            outs = model(imgs)
        torch.cuda.synchronize()
        t_total += time.perf_counter() - t0
        for (stem, _), o in zip(chunk, outs):
            b = o["boxes"].float().cpu().numpy()
            preds[stem] = [[int(l) + model.label_offset - 1, float(s), float(x1), float(y1), float(x2 - x1), float(y2 - y1)]
                           for (x1, y1, x2, y2), s, l in zip(b, o["scores"].float().cpu().numpy(),
                                                             o["labels"].cpu().numpy())]
    return preds, 1000 * t_total / len(items)


def load_trained(a):
    saved = json.load(open(RUNS / a.name / "args.json"))
    for k in ["arch", "anchors", "ratios", "max_size", "rpn_pre", "rpn_post", "roi_batch", "roi_pos",
              "box_loss", "focal_alpha", "focal_gamma", "iou_pos", "iou_neg"]:
        setattr(a, k, saved[k])
    model = build(a).to(DEVICE)
    model.load_state_dict(torch.load(RUNS / a.name / "last.pth", map_location=DEVICE))
    return model


def eval_run(a):
    model = load_trained(a)
    preds, ms = predict(model, val_items())
    PRED.mkdir(exist_ok=True)
    json.dump(preds, open(PRED / f"val_{a.name}.json", "w"))
    m = evaluate(preds, PREP / "val_s5" / "coco_gt.json")
    m.update(run=a.name, imgsz=a.max_size, ms_per_img=round(ms, 1))
    RESULTS.mkdir(exist_ok=True)
    with open(RESULTS / "val_metrics.jsonl", "a") as f:
        f.write(json.dumps(m) + "\n")
    print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in m.items()}, indent=1))


def test_run(a):
    model = load_trained(a)
    preds, _ = predict(model, test_items())
    json.dump(preds, open(PRED / f"test_{a.name}.json", "w"))
    to_submission(preds, Path("submissions") / f"{a.name}.csv")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["train", "eval", "test"])
    ap.add_argument("--arch", choices=["frcnn", "retina"], default="frcnn")
    ap.add_argument("--name", required=True)
    ap.add_argument("--train", default="train_s10")
    ap.add_argument("--max-size", type=int, default=1024)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--eval-every", type=int, default=3)
    ap.add_argument("--limit", type=int, default=0, help="debug: use only N train / val images")
    # anchors (both archs): base sizes per FPN level; None = torchvision default (32..512)
    ap.add_argument("--anchors", type=int, nargs="*", default=None)
    ap.add_argument("--ratios", type=float, nargs="*", default=[0.5, 1.0, 2.0])
    # Faster R-CNN
    ap.add_argument("--rpn-pre", type=int, default=2000)
    ap.add_argument("--rpn-post", type=int, default=1000)
    ap.add_argument("--roi-batch", type=int, default=512)
    ap.add_argument("--roi-pos", type=float, default=0.25)
    ap.add_argument("--box-loss", default="default", help="frcnn: default|giou; retina: default(l1)|giou|ciou|diou|smooth_l1")
    # RetinaNet
    ap.add_argument("--focal-alpha", type=float, default=0.25)
    ap.add_argument("--focal-gamma", type=float, default=2.0)
    ap.add_argument("--iou-pos", type=float, default=0.5)
    ap.add_argument("--iou-neg", type=float, default=0.4)
    a = ap.parse_args()
    {"train": train, "eval": eval_run, "test": test_run}[a.mode](a)
