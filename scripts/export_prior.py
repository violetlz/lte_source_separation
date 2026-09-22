#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rfsep.priors.inference import export_prior_for_inference, load_prior_for_inference  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--source", choices=("target", "interference"), required=True)
    parser.add_argument("--weights", choices=("raw", "ema"), default="ema")
    args = parser.parse_args()
    digest = export_prior_for_inference(args.checkpoint, args.output,
                                        source=args.source, weights=args.weights)
    reloaded = load_prior_for_inference(args.output, source=args.source,
                                        device="cpu", weights=args.weights)
    result = {"output": str(Path(args.output).resolve()), "sha256": digest,
              "source": reloaded.source, "weights": reloaded.weights,
              "global_step": reloaded.global_step,
              "all_parameters_frozen": all(not p.requires_grad for p in reloaded.model.parameters())}
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
