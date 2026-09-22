#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rfsep.config import load_platform  # noqa: E402
from rfsep.contracts import Observation, complex_to_network, network_to_complex  # noqa: E402
from rfsep.evaluation.metrics import bit_errors, nmse  # noqa: E402
from rfsep.observation import build_example, load_calibration  # noqa: E402
from rfsep.priors.inference import load_prior_for_inference  # noqa: E402
from rfsep.separation.rgs import RGSConfig, separate_rgs  # noqa: E402
from rfsep.waveforms.constellations import demap_symbols, map_bits  # noqa: E402
from rfsep.waveforms.ofdm import demodulate_waveform, modulate_grid  # noqa: E402
from rfsep.waveforms.receiver import receive_target  # noqa: E402


def _direct_initial(y: np.ndarray, platform) -> np.ndarray:
    outputs = []
    for slot in range(2):
        half = y[slot * 15360:(slot + 1) * 15360]
        grid = demodulate_waveform(half, platform.target_subcarriers,
                                   platform.fft_size, platform.cp_lengths[:7])
        bits = demap_symbols(grid, "QPSK").reshape(7, 300, 2)
        hard = map_bits(bits, "QPSK")
        outputs.append(modulate_grid(hard, platform.target_subcarriers,
                                     platform.fft_size, platform.cp_lengths[:7],
                                     dtype=np.complex64))
    return np.concatenate(outputs)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-checkpoint", required=True)
    parser.add_argument("--interference-checkpoint", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    platform = load_platform(ROOT / "configs/platform/lte_v1.yaml")
    calibration = load_calibration(ROOT / "reports/formal_validation/calibration_summary.json", platform)
    target = load_prior_for_inference(args.target_checkpoint, source="target",
                                      device=args.device, weights="ema")
    interference = load_prior_for_inference(args.interference_checkpoint, source="interference",
                                            device=args.device, weights="ema")
    cases, mixtures, initials = [], [], []
    truths = []
    for parent_id in range(2):
        for sir in (-6, 0, 6):
            observation, info, truth = build_example(platform, "audit", 1000000 + parent_id,
                                                     sir, calibration, dtype=np.complex64)
            direct = _direct_initial(observation.y, platform)
            for slot in (0, 1):
                begin = slot * 15360
                mixtures.append(complex_to_network(observation.y[begin:begin + 15360]))
                initials.append(complex_to_network(direct[begin:begin + 15360]))
                truths.append(truth.target_waveform[begin:begin + 15360])
                cases.append((parent_id, sir, slot, info.kappa, truth.target_bits))
    # kappa differs by SIR, so batch same-kappa cases while keeping one public API.
    rows = []
    for sir in (-6, 0, 6):
        indices = [index for index, case in enumerate(cases) if case[1] == sir]
        y = torch.from_numpy(np.stack([mixtures[index] for index in indices])).to(args.device)
        initial = torch.from_numpy(np.stack([initials[index] for index in indices])).to(args.device)
        kappa = cases[indices[0]][3]
        estimate, audit = separate_rgs(y, kappa, initial, target, interference,
                                       RGSConfig(iterations=16, max_lr=0.005,
                                                 min_lr=1e-6, seed=2026092305 + sir))
        moved = torch.sqrt(torch.mean((estimate - initial) ** 2, dim=(1, 2))).cpu().numpy()
        for local, index in enumerate(indices):
            estimated = network_to_complex(estimate[local].cpu().numpy())
            if not np.isfinite(estimated).all() or moved[local] <= 0:
                raise RuntimeError("RGS smoke output is not finite or did not update")
            rows.append({"parent": cases[index][0], "sir_re_db": sir, "slot": cases[index][2],
                         "kappa": kappa, "update_rms": float(moved[local]),
                         "target_nmse": nmse(estimated, truths[index]),
                         "reconstruction_max_abs": audit["reconstruction_max_abs"]})
    payload = {"state": "passed", "engineering_only": True,
               "target_step": target.global_step, "interference_step": interference.global_step,
               "weights": "ema", "iterations": 16, "cases": rows,
               "truth_entered_in_inference": False,
               "all_parameters_frozen": all(not p.requires_grad for p in target.model.parameters())
                                        and all(not p.requires_grad for p in interference.model.parameters())}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
