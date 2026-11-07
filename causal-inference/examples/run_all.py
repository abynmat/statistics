"""Run every example in order and regenerate all figures/ and results/.

Author: Abyn Scaria. MIT License.

Usage (from the repo root or from examples/):
    python3 examples/run_all.py                      # everything
    python3 examples/run_all.py ex05_rdd ex08_iv     # a subset

Each example runs in its own Python process. Running them in one interpreter
made ex11 about 70x slower after ex10's joblib process pool (the pool is kept
alive for reuse), so isolation is both cleaner and faster.
Total wall clock on an Apple-silicon laptop: about 3 minutes.
"""
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent

EXAMPLES = [
    "ex01_confounding", "ex02_bad_controls", "ex03_dags_frontdoor", "ex04_ab_testing",
    "ex05_rdd", "ex06_did_staggered", "ex07_synthetic_control", "ex08_iv",
    "ex09_hte_metalearners", "ex10_dml", "ex11_causal_discovery", "ex12_sensitivity",
]

if __name__ == "__main__":
    only = set(sys.argv[1:])
    total = time.time()
    for name in EXAMPLES:
        if only and name not in only:
            continue
        print(f"\n===== {name} =====", flush=True)
        t0 = time.time()
        subprocess.run([sys.executable, "-u", str(HERE / f"{name}.py")], cwd=HERE, check=True)
        print(f"[{name}] done in {time.time() - t0:.1f}s", flush=True)
    print(f"\nall done in {time.time() - total:.1f}s")
