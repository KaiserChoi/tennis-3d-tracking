"""Run all accuracy evaluations and report PASS/FAIL.

Usage:
    python -m tools.run_accuracy_loop [--max-frames 1800]
"""

import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

THRESHOLDS = {
    "bounce_mean_cm": 30.0,
    "bounce_recall": 0.80,
    "speed_pass_rate": 80.0,
}


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-frames", type=int, default=1800)
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("  ACCURACY ITERATION LOOP")
    print("=" * 60)

    # 1. Bounce accuracy
    print("\n>>> Running bounce accuracy evaluation...")
    from tools.eval_bounce_accuracy import main as run_bounce
    import sys
    sys.argv = ["eval_bounce_accuracy", f"--max-frames={args.max_frames}"]
    bounce_results = run_bounce()

    # 2. Speed accuracy
    print("\n>>> Running speed accuracy evaluation...")
    sys.argv = ["eval_speed_accuracy", f"--max-frames={args.max_frames}"]
    from tools.eval_speed_accuracy import main as run_speed
    speed_results = run_speed()

    # Summary
    print("\n" + "=" * 60)
    print("  OVERALL SUMMARY")
    print("=" * 60)

    results = []

    # Bounce mean
    if bounce_results and bounce_results.get("mean_cm") is not None:
        v = bounce_results["mean_cm"]
        t = THRESHOLDS["bounce_mean_cm"]
        ok = v < t
        results.append(ok)
        print(f"  Bounce mean error:   {v:6.1f} cm  (threshold < {t} cm)  {'PASS ✓' if ok else 'FAIL ✗'}")
    else:
        results.append(False)
        print(f"  Bounce mean error:   N/A  FAIL ✗")

    # Bounce recall
    if bounce_results and bounce_results.get("recall") is not None:
        v = bounce_results["recall"]
        t = THRESHOLDS["bounce_recall"]
        ok = v >= t
        results.append(ok)
        print(f"  Bounce recall:       {v*100:5.1f}%   (threshold >= {t*100:.0f}%)  {'PASS ✓' if ok else 'FAIL ✗'}")
    else:
        results.append(False)
        print(f"  Bounce recall:       N/A  FAIL ✗")

    # Speed
    if speed_results and speed_results.get("pass_rate") is not None:
        v = speed_results["pass_rate"]
        t = THRESHOLDS["speed_pass_rate"]
        ok = v >= t
        results.append(ok)
        print(f"  Speed pass rate:     {v:5.1f}%   (threshold >= {t:.0f}%)  {'PASS ✓' if ok else 'FAIL ✗'}")
    else:
        results.append(False)
        print(f"  Speed pass rate:     N/A  FAIL ✗")

    all_pass = all(results)
    print(f"\n  OVERALL: {'ALL PASS ✓✓✓' if all_pass else 'SOME FAILED — iterate and rerun'}")
    print("=" * 60 + "\n")

    return 0 if all_pass else 1


if __name__ == "__main__":
    exit(main())
