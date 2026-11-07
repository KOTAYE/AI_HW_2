#!/usr/bin/env bash
# Push a Kaggle GPU kernel that runs one benchmark group: bash kaggle/push.sh frcnn
set -e
GROUP=$1
USER_NAME=${KAGGLE_USER:-vitya001}
D=$(mktemp -d)
sed "s/__GROUP__/$GROUP/" "$(dirname "$0")/run_group.py" > "$D/run_group.py"
cat > "$D/kernel-metadata.json" <<EOF
{
  "id": "$USER_NAME/hw2-bench-$GROUP",
  "title": "hw2-bench-$GROUP",
  "code_file": "run_group.py",
  "language": "python",
  "kernel_type": "script",
  "is_private": "true",
  "enable_gpu": "true",
  "enable_internet": "true",
  "competition_sources": ["multiple-object-video-detection-hw-2-ucu-ai-2026"]
}
EOF
kaggle kernels push -p "$D" --accelerator NvidiaTeslaT4
