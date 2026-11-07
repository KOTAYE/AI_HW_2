#!/usr/bin/env bash
# Upload the current code as the private Kaggle dataset `hw2-code` and push a GPU kernel that runs one
# benchmark group (Kaggle kernels have no internet here):   bash kaggle/push.sh frcnn [--no-upload]
set -e
cd "$(dirname "$0")/.."
GROUP=$1
USER_NAME=${KAGGLE_USER:-vitya001}
if [ "$2" != "--no-upload" ]; then
  C=kaggle/_upload; rm -rf $C; mkdir -p $C
  tar -czf "$C/code.tar.gz" src scripts models
  echo "{\"title\": \"hw2-code\", \"id\": \"$USER_NAME/hw2-code\", \"licenses\": [{\"name\": \"CC0-1.0\"}]}" > "$C/dataset-metadata.json"
  if kaggle datasets status "$USER_NAME/hw2-code" >/dev/null 2>&1; then
    (cd "$C" && kaggle datasets version -p . -m "update" -q)
  else
    (cd "$C" && kaggle datasets create -p . -q)
  fi
  until kaggle datasets status "$USER_NAME/hw2-code" 2>&1 | grep -q ready; do sleep 5; done
fi
D=kaggle/_kernel; rm -rf $D; mkdir -p $D
sed "s/__GROUP__/$GROUP/" kaggle/run_group.py > "$D/run_group.py"
cat > "$D/kernel-metadata.json" <<META
{
  "id": "$USER_NAME/hw2-bench-$GROUP",
  "title": "hw2-bench-$GROUP",
  "code_file": "run_group.py",
  "language": "python",
  "kernel_type": "script",
  "is_private": "true",
  "enable_gpu": "true",
  "enable_internet": "false",
  "dataset_sources": ["$USER_NAME/hw2-code"],
  "competition_sources": ["multiple-object-video-detection-hw-2-ucu-ai-2026"]
}
META
(cd "$D" && kaggle kernels push -p . --accelerator NvidiaTeslaT4)
