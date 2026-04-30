"""
run_all.py
==========
Single entry-point that runs the full Dynamic Perceptual Steering pipeline
in sequence: bootstrap → phase2 → phase3a → phase3b → phase3c → phase5.

Phase 4 (adversarial) is opt-in because it requires manually placed images.

Usage:
    python run_all.py                              # full pipeline
    python run_all.py --config configs/config.yaml
    python run_all.py --skip-apo                   # skip the 12h APO phase
    python run_all.py --phases 2 3a 5              # run specific phases only
    python run_all.py --include-phase4             # also run adversarial phase
"""

import argparse
import logging
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LOG_DIR = ROOT / "results" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_DIR / "run_all.log", mode="a"),
    ],
)
logger = logging.getLogger("run_all")

PHASES = {
    "bootstrap": ("scripts/bootstrap_data_layout.py", None),
    "2":         ("experiments/phase2_baseline.py",        "~4 h"),
    "3a":        ("experiments/phase3a_manual_steering.py", "~6 h"),
    "3b":        ("experiments/phase3b_apo.py",             "~12 h"),
    "3c":        ("experiments/phase3c_probing.py",         "~8 h"),
    "4":         ("experiments/phase4_adversarial.py",      "~2 h"),
    "5":         ("experiments/phase5_final_analysis.py",   "~5 min"),
}


def run_phase(name: str, script: str, config: str, eta: str = None):
    label = f"Phase {name}" if name != "bootstrap" else "Bootstrap"
    eta_str = f"  (estimated {eta})" if eta else ""
    logger.info("")
    logger.info("=" * 60)
    logger.info(f"STARTING  {label}{eta_str}")
    logger.info("=" * 60)

    cmd = [sys.executable, str(ROOT / script)]
    if name != "bootstrap":
        cmd += ["--config", config]

    phase_log = LOG_DIR / f"phase{name}.log"
    t0 = time.time()

    with open(phase_log, "a") as log_fh:
        process = subprocess.Popen(
            cmd,
            cwd=str(ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        for line in process.stdout:
            print(line, end="", flush=True)
            log_fh.write(line)
            log_fh.flush()
        process.wait()

    elapsed = (time.time() - t0) / 60
    if process.returncode == 0:
        logger.info(f"COMPLETED {label} in {elapsed:.1f} min  →  log: {phase_log}")
        return True
    else:
        logger.error(f"FAILED    {label} after {elapsed:.1f} min  (exit {process.returncode})")
        logger.error(f"Full log: {phase_log}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Run the full DPS pipeline")
    parser.add_argument("--config", default="configs/config.yaml",
                        help="Path to config YAML")
    parser.add_argument("--skip-apo", action="store_true",
                        help="Skip Phase 3B (APO) — saves ~12 h")
    parser.add_argument("--include-phase4", action="store_true",
                        help="Include Phase 4 adversarial (needs images in data/raw/adversarial/)")
    parser.add_argument("--phases", nargs="+", metavar="PHASE",
                        help="Run only specific phases, e.g. --phases 2 3a 5")
    parser.add_argument("--stop-on-error", action="store_true",
                        help="Abort the pipeline if any phase fails")
    args = parser.parse_args()

    # Determine which phases to run
    default_sequence = ["bootstrap", "2", "3a", "3b", "3c", "5"]
    if args.skip_apo:
        default_sequence.remove("3b")
    if args.include_phase4:
        default_sequence.insert(default_sequence.index("5"), "4")

    sequence = args.phases if args.phases else default_sequence

    logger.info("=" * 60)
    logger.info("DYNAMIC PERCEPTUAL STEERING — FULL PIPELINE")
    logger.info(f"Config : {args.config}")
    logger.info(f"Phases : {' → '.join(sequence)}")
    logger.info("=" * 60)

    results = {}
    pipeline_start = time.time()

    for phase in sequence:
        if phase not in PHASES:
            logger.warning(f"Unknown phase '{phase}' — skipping.")
            continue
        script, eta = PHASES[phase]
        ok = run_phase(phase, script, args.config, eta)
        results[phase] = ok
        if not ok and args.stop_on_error:
            logger.error("Pipeline aborted (--stop-on-error).")
            break

    total_min = (time.time() - pipeline_start) / 60
    logger.info("")
    logger.info("=" * 60)
    logger.info(f"PIPELINE COMPLETE  ({total_min:.0f} min total)")
    logger.info("=" * 60)
    for phase, ok in results.items():
        status = "✓" if ok else "✗ FAILED"
        logger.info(f"  Phase {phase:<10} {status}")
    logger.info("")
    logger.info(f"Results : {ROOT / 'results'}/")
    logger.info(f"Figures : {ROOT / 'results' / 'figures'}/")
    logger.info(f"Logs    : {LOG_DIR}/")

    failed = [p for p, ok in results.items() if not ok]
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
