#!/usr/bin/env bash
# Required model comparison under one fixed protocol:
#   train = data/prepared/train_s10 (every 10th frame, ignored regions painted), long side 1024,
#   val   = data/prepared/val_s5 (every 5th frame of the 7 val sequences), COCO metrics via src/coco_eval.py.
# Each run appends a line to results/val_metrics.jsonl. Usage: bash scripts/run_benchmarks.sh [group]
set -e
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8
mkdir -p logs
run() { local name=$1; shift; echo "=== $name"; "$@" > "logs/$name.log" 2>&1; }

frcnn() {   # two-stage anchor-based (torchvision Faster R-CNN R50-FPN v2), 12 epochs
  run frcnn_base   python src/tv.py train --arch frcnn --name frcnn_base
  run frcnn_base_e python src/tv.py eval  --arch frcnn --name frcnn_base
  # anchors one octave smaller (EDA: 45% of GT have no anchor with IoU>=0.5 at default 32-512)
  run frcnn_anc16   python src/tv.py train --arch frcnn --name frcnn_anc16 --anchors 16 32 64 128 256
  run frcnn_anc16_e python src/tv.py eval  --arch frcnn --name frcnn_anc16
  # + more proposals for dense scenes (up to ~180 objects/frame)
  run frcnn_anc16_rpn   python src/tv.py train --arch frcnn --name frcnn_anc16_rpn --anchors 16 32 64 128 256 --rpn-pre 4000 --rpn-post 2000 --roi-batch 1024
  run frcnn_anc16_rpn_e python src/tv.py eval  --arch frcnn --name frcnn_anc16_rpn
  # + GIoU box regression loss in the second stage (loss modification)
  run frcnn_anc16_giou   python src/tv.py train --arch frcnn --name frcnn_anc16_giou --anchors 16 32 64 128 256 --box-loss giou
  run frcnn_anc16_giou_e python src/tv.py eval  --arch frcnn --name frcnn_anc16_giou
}

retina() {  # one-stage anchor-based (torchvision RetinaNet R50-FPN v2), 12 epochs
  run retina_base   python src/tv.py train --arch retina --name retina_base
  run retina_base_e python src/tv.py eval  --arch retina --name retina_base
  run retina_anc16   python src/tv.py train --arch retina --name retina_anc16 --anchors 16 32 64 128 256
  run retina_anc16_e python src/tv.py eval  --arch retina --name retina_anc16
  # GIoU regression loss + lower positive IoU threshold for tiny boxes
  run retina_anc16_giou   python src/tv.py train --arch retina --name retina_anc16_giou --anchors 16 32 64 128 256 --box-loss giou --iou-pos 0.4 --iou-neg 0.3
  run retina_anc16_giou_e python src/tv.py eval  --arch retina --name retina_anc16_giou
}

yolo() {    # one-stage anchor-free (Ultralytics YOLO11s), 30 epochs
  run yolo11s_1024_e python src/ultra.py eval --name yolo11s_1024 --imgsz 1024
  # extra stride-4 (P2) detection level
  run yolo11s_p2_1024   python src/ultra.py train --model yolo11s.pt --cfg-yaml models/yolo11s-p2.yaml --name yolo11s_p2_1024 --imgsz 1024 --batch 4
  run yolo11s_p2_1024_e python src/ultra.py eval  --name yolo11s_p2_1024 --imgsz 1024
  # higher input resolution
  run yolo11s_1280   python src/ultra.py train --model yolo11s.pt --name yolo11s_1280 --imgsz 1280 --batch 4
  run yolo11s_1280_e python src/ultra.py eval  --name yolo11s_1280 --imgsz 1280
  # loss re-weighting: stronger box / DFL terms (localisation of tiny boxes)
  run yolo11s_1024_box   python src/ultra.py train --model yolo11s.pt --name yolo11s_1024_box --imgsz 1024 --batch 4 --set box=10.0 dfl=2.0
  run yolo11s_1024_box_e python src/ultra.py eval  --name yolo11s_1024_box --imgsz 1024
}

detr() {    # query-based (Ultralytics RT-DETR-L), same wall-clock budget as YOLO
  run rtdetr_l_1024   python src/ultra.py train --model rtdetr-l.pt --name rtdetr_l_1024 --imgsz 1024 --batch 4 --epochs 15
  run rtdetr_l_1024_e python src/ultra.py eval  --name rtdetr_l_1024 --imgsz 1024
  run rtdetr_l_q500   python src/ultra.py train --model rtdetr-l.pt --cfg-yaml models/rtdetr-l-q500.yaml --name rtdetr_l_q500 --imgsz 1024 --batch 4 --epochs 15
  run rtdetr_l_q500_e python src/ultra.py eval  --name rtdetr_l_q500 --imgsz 1024
}

groups=("$@"); [ ${#groups[@]} -eq 0 ] && groups=(frcnn retina yolo detr)
for g in "${groups[@]}"; do $g; done
