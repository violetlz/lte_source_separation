from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.io import loadmat

from ..config import load_platform
from ..waveforms.ofdm import modulate_grid


def crosscheck(reference_path: str | Path, config_path: str | Path,
               output_path: str | Path) -> dict:
    data = loadmat(reference_path, simplify_cells=True)
    config = load_platform(config_path)
    result = {
        "matlab_version": str(data["matlab_version"]),
        "lte_toolbox_version": str(data["lte_toolbox_version"]),
        "platform_hash": config.config_hash,
        "grid_alignment": "MATLAB [subcarrier,symbol] explicitly transposed to Python [symbol,subcarrier]",
        "scale_rule": "MATLAB waveform = Python fixed-unit-power waveform * sqrt(K)/N",
    }
    for role, tones in (("target", config.target_subcarriers),
                        ("interference", config.interference_subcarriers)):
        grid = np.asarray(data[f"{role}_grid"]).T
        python_waveform = modulate_grid(grid, tones, config.fft_size, config.cp_lengths)
        matlab_waveform = np.asarray(data[f"{role}_waveform"]).reshape(-1)
        expected_scale = np.sqrt(len(tones)) / config.fft_size
        reconstructed_matlab = python_waveform * expected_scale
        relative_error = float(np.linalg.norm(matlab_waveform - reconstructed_matlab)
                               / np.linalg.norm(matlab_waveform))
        info = data[f"{role}_info"]
        result[f"{role}_relative_error"] = relative_error
        result[f"{role}_expected_scale"] = float(expected_scale)
        result[f"{role}_sampling_rate"] = int(info["SamplingRate"])
        result[f"{role}_nfft"] = int(info["Nfft"])
        result[f"{role}_windowing"] = int(info["Windowing"])
        result[f"{role}_cp_lengths"] = [int(x) for x in np.asarray(info["CyclicPrefixLengths"]).reshape(-1)]
    result["passed"] = bool(
        result["target_relative_error"] < 1e-12
        and result["interference_relative_error"] < 1e-12
        and result["target_sampling_rate"] == result["interference_sampling_rate"] == 30_720_000
        and result["target_nfft"] == result["interference_nfft"] == 2048
        and result["target_windowing"] == result["interference_windowing"] == 0
        and result["target_cp_lengths"] == result["interference_cp_lengths"] == list(config.cp_lengths)
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = crosscheck(args.reference, args.config, args.output)
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
