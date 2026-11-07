#!/usr/bin/env bash
# Local GPU queue: wait for the running YOLO baseline, then evaluate it and run the remaining benchmarks.
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8
while python -c "import psutil,sys; sys.exit(0 if any('ultra.py train' in ' '.join(p.info['cmdline'] or []) for p in psutil.process_iter(['cmdline'])) else 1)"; do sleep 30; done
python src/ultra.py eval --name yolo11s_1024 --imgsz 1024 > logs/yolo11s_1024_e.log 2>&1
python src/ultra.py test --name yolo11s_1024 --imgsz 1024 > logs/yolo11s_1024_t.log 2>&1
bash scripts/run_benchmarks.sh frcnn retina yolo detr
