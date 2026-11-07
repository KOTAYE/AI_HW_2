"""Kaggle kernel entry point: runs one benchmark group of scripts/run_benchmarks.sh on a Kaggle GPU.

The kernel clones this (public) repository, builds the same prepared subsets from the competition data
and copies run logs / weights / predictions / metrics to /kaggle/working/out for download.
The group name is substituted by kaggle/push.sh before pushing.
"""
import glob
import os
import shutil
import subprocess

GROUP = "__GROUP__"
REPO = "https://github.com/KOTAYE/AI_HW_2.git"
W = "/kaggle/working"


def sh(cmd: str) -> None:
    print("+", cmd, flush=True)
    subprocess.run(cmd, shell=True, check=True)


sh(f"git clone --depth 1 {REPO} {W}/repo")
sh("pip install -q faster-coco-eval")
data = os.path.dirname(glob.glob("/kaggle/input/**/test_images.csv", recursive=True)[0])
os.environ.update(HW2_DATA=data, HW2_PREP=f"{W}/prepared", PYTHONIOENCODING="utf-8")
os.chdir(f"{W}/repo")
sh("nvidia-smi")
sh("python src/prepare_data.py --split val --step 5 --workers 4")
sh("python src/prepare_data.py --split train --step 10 --workers 4")
try:
    sh(f"bash scripts/run_benchmarks.sh {GROUP}")
finally:
    out = f"{W}/out"
    for d in ["runs", "predictions", "results", "logs"]:
        if os.path.isdir(d):
            shutil.copytree(d, f"{out}/{d}", dirs_exist_ok=True,
                            ignore=shutil.ignore_patterns("*.jpg", "*.png", "labels.cache"))
    shutil.rmtree(f"{W}/prepared", ignore_errors=True)
    shutil.rmtree(f"{W}/repo", ignore_errors=True)
