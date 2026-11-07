"""Run the test suite and every chapter script, regenerating figures/ and results/.

Author: Abyn Scaria. MIT License.

Usage (from the repo root):
    python3 examples/run_all.py                 # tests + all chapters
    python3 examples/run_all.py ch03 ch07       # a subset (prefix match)

Each script runs in its own process so timings in one chapter are not affected by another.
"""
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

CHAPTERS = [
    "ch01_grid_and_conjugate",
    "ch02_metropolis_hastings",
    "ch03_hamiltonian_monte_carlo",
    "ch04_bayesian_ab_testing",
    "ch05_hierarchical_partial_pooling",
    "ch06_bayesian_regression",
    "ch07_kalman_filter",
    "ch08_particle_filter",
    "ch09_model_comparison",
]

if __name__ == "__main__":
    only = sys.argv[1:]
    total = time.time()
    if not only:
        print("===== tests =====", flush=True)
        subprocess.run([sys.executable, "-u", str(ROOT / "tests" / "test_bayeslab.py")], cwd=ROOT, check=True)
    timings = {}
    for name in CHAPTERS:
        if only and not any(name.startswith(o) for o in only):
            continue
        print(f"\n===== {name} =====", flush=True)
        t0 = time.time()
        subprocess.run([sys.executable, "-u", str(HERE / f"{name}.py")], cwd=HERE, check=True)
        timings[name] = time.time() - t0
        print(f"[{name}] done in {timings[name]:.1f}s", flush=True)
    print("\nwall clock per chapter:")
    for k, v in timings.items():
        print(f"  {k:36s} {v:6.1f}s")
    print(f"all done in {time.time() - total:.1f}s")
