#!/usr/bin/env bash
# Local GPU queue: evaluate the YOLO baseline, make its submission, then run the remaining benchmarks.
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8
python src/ultra.py eval --name yolo11s_1024 --imgsz 1024 > logs/yolo11s_1024_e.log 2>&1
python src/ultra.py test --name yolo11s_1024 --imgsz 1024 > logs/yolo11s_1024_t.log 2>&1
bash scripts/run_benchmarks.sh frcnn retina yolo detr
