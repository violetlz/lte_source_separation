#!/usr/bin/env python3
"""Render the audit BER curves and paired RGS-minus-Direct differences."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("summary_dir", type=Path)
    args = parser.parse_args()
    summary = read_csv(args.summary_dir / "summary.csv")
    differences = read_csv(args.summary_dir / "paired_differences.csv")
    sirs = sorted({int(row["sir_re_db"]) for row in summary})

    def series(method: str, budget: int = 0) -> list[float]:
        selected = {int(row["sir_re_db"]): float(row["ber"]) for row in summary
                    if row["method"] == method and
                    (row["trajectory_kind"] == "baseline" if budget == 0 else
                     row["trajectory_kind"] == "completed_budget" and
                     int(row["schedule_total_steps"]) == budget)}
        if set(selected) != set(sirs):
            raise ValueError(f"incomplete BER curve for {method}, {budget}")
        return [selected[sir] for sir in sirs]

    fig, (ax_ber, ax_gain) = plt.subplots(2, 1, figsize=(9.4, 8.2), sharex=True,
                                         gridspec_kw={"height_ratios": [1.3, 1]})
    for method, label, style in (
        ("direct", "Direct", {"color": "#252525", "marker": "o", "linestyle": "--"}),
        ("fft_diagonal_wiener_approx", "FFT-diagonal Wiener approx.",
         {"color": "#9c622a", "marker": "s", "linestyle": ":"}),
        ("single_re_marginal_bit_map_estimated", "Estimated single-RE bit-MAP",
         {"color": "#633d91", "marker": "D", "linestyle": "-"}),
    ):
        ax_ber.plot(sirs, series(method), label=label, linewidth=1.8, markersize=5, **style)
    colors = {1000: "#1d78a5", 2000: "#258a68", 5000: "#bc3e3e"}
    for budget in (1000, 2000, 5000):
        ax_ber.plot(sirs, series("waveform_alpha_rgs", budget), label=f"RGS {budget} completed",
                    color=colors[budget], marker=".", linewidth=1.2, markersize=8,
                    alpha=0.75)
        rows = {int(row["sir_re_db"]): row for row in differences
                if int(row["rgs_completed_budget"]) == budget and row["reference"] == "direct"}
        if set(rows) != set(sirs):
            raise ValueError(f"incomplete paired Direct difference for {budget}")
        values = [1000 * float(rows[sir]["mean_ber_difference"]) for sir in sirs]
        lower = [1000 * float(rows[sir]["paired_bootstrap_95_lower"]) for sir in sirs]
        upper = [1000 * float(rows[sir]["paired_bootstrap_95_upper"]) for sir in sirs]
        ax_gain.plot(sirs, values, color=colors[budget], marker="o", linewidth=1.5,
                     markersize=4, label=f"RGS {budget} - Direct")
        ax_gain.fill_between(sirs, lower, upper, color=colors[budget], alpha=0.13)

    ax_ber.set_ylabel("Bit error rate")
    ax_ber.set_ylim(0, 0.51)
    ax_ber.grid(alpha=0.25)
    ax_ber.legend(loc="upper right", fontsize=8, ncol=2)
    ax_ber.set_title("LTE waveform RGS paired budget audit (16 audit parents; 134,400 bits/SIR)")
    ax_gain.axhline(0, color="black", linewidth=0.8)
    ax_gain.set_ylabel("BER difference vs Direct (×10⁻³)\nnegative favors RGS")
    ax_gain.set_xlabel("SIR_RE (dB)")
    ax_gain.set_xticks(sirs)
    ax_gain.grid(alpha=0.25)
    ax_gain.legend(loc="lower right", fontsize=8)
    fig.text(0.5, 0.01, "Paired parent bootstrap 95% intervals; audit split only. "
             "Wiener and bit-MAP are estimated approximations, not exact optimum detectors.",
             ha="center", va="bottom", fontsize=8)
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    for suffix in ("png", "pdf"):
        fig.savefig(args.summary_dir / f"ber_budget_audit.{suffix}", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
