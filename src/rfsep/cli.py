from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import load_experiment, load_platform
from .diagnostics.waveform_audit import run_waveform_audit
from .evaluation.reporting import write_validation_report
from .evaluation.runner import evaluate
from .observation import calibrate


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _resolve_run(root: Path, value: str) -> Path:
    candidate = Path(value)
    if candidate.is_dir():
        return candidate.resolve()
    candidate = root / "artifacts" / "runs" / value
    if candidate.is_dir():
        return candidate.resolve()
    raise FileNotFoundError(f"run not found: {value}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rfsep")
    sub = parser.add_subparsers(dest="command", required=True)
    audit_parser = sub.add_parser("audit", help="run deterministic waveform audits")
    audit_parser.add_argument("--config", required=True)
    audit_parser.add_argument("--output", default="artifacts/runs/waveform_audit.json")
    calibration_parser = sub.add_parser("calibrate", help="estimate independent in-band interference power")
    calibration_parser.add_argument("--config", required=True)
    calibration_parser.add_argument("--num-records", type=int, default=4096)
    calibration_parser.add_argument("--output", default="artifacts/calibration/lte_v1_calibration.json")
    calibration_parser.add_argument("--provisional", action="store_true")
    evaluate_parser = sub.add_parser("evaluate", help="run Direct and audit receivers")
    evaluate_parser.add_argument("--config", required=True)
    report_parser = sub.add_parser("report", help="render the validation report for a run")
    report_parser.add_argument("--run", required=True)
    return parser


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    root = _project_root()
    if args.command == "audit":
        result = run_waveform_audit(load_platform(args.config), root / args.output)
        print(json.dumps({"all_passed": result["all_passed"], "passed": result["passed"],
                          "failed": result["failed"], "output": str((root / args.output).resolve())}, indent=2))
        if not result["all_passed"]:
            raise SystemExit(1)
    elif args.command == "calibrate":
        payload = calibrate(load_platform(args.config), args.num_records, root / args.output,
                            provisional=args.provisional)
        print(json.dumps(payload, indent=2, sort_keys=True))
    elif args.command == "evaluate":
        experiment = load_experiment(args.config)
        run_dir = evaluate(experiment, root)
        report = None
        if experiment.raw["name"] == "platform_validation":
            report = write_validation_report(root, run_dir)
        print(json.dumps({"run_dir": str(run_dir), "report": str(report) if report else None}, indent=2))
    elif args.command == "report":
        run_dir = _resolve_run(root, args.run)
        print(write_validation_report(root, run_dir))
