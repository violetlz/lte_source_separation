# Why waveform RGS NMSE does not become BER gain

This is a diagnosis of the frozen, no-noise LTE waveform RGS implementation,
**not a fix or a new formal result**. It uses only audit parents
`1000000..1000003` at SIR_RE `[-24,-6,0,6]`: 16 recomputed 5,000-step
trajectories, 33,600 bits per SIR. The final target/interference EMA weights,
initialization, LR, schedule, slot seeds, and receiver are identical to the
paired budget audit. Every recomputed Direct/RGS error count and RGS waveform
NMSE matches its original atomic audit shard. No partial sealed-test RGS
result was read for diagnosis or tuning.

## The actual decision chain

```text
mixture y ── CP removal + FFT on target tones ── hard QPSK ── Direct bits
             │                                │
             │                       remodulate exact QPSK+CP waveform s₀
             │                                │
             └──────────── y, κ ─────────── waveform RGS ── ŝ
                                                     │
                                      same CP/FFT/QPSK receiver ── RGS bits
```

The hard-remodulated starting waveform `s₀` has exactly the **same bits** as
Direct. It is already a valid target QPSK/OFDM waveform, with target tones and
CP and essentially zero energy on other useful FFT tones. RGS can improve BER
only if a target RE's real or imaginary component crosses the zero decision
axis. Reducing squared waveform or grid error without crossing an axis cannot
change a hard bit. The source-code path and the independent recomputation
confirmed the intended complex-coordinate conversion and receiver alignment;
no bit-order, slot-concatenation, or denominator mismatch was found.

## Recomputed audit evidence

| SIR_RE dB | Direct/RGS errors (of 33,600) | Bit flips | Corrected / newly wrong | Waveform NMSE: mixture → hard init → RGS | Target-grid NMSE: hard init → RGS |
|---:|---:|---:|---:|---:|---:|
| -24 | 16,016 / 15,992 | 90 | 57 / 33 | 1008.69 → 1.909 → **7.874** | 1.907 → **5.647** |
| -6 | 11,383 / 11,383 | **0** | 0 / 0 | 15.987 → 1.355 → 1.301 | 1.355 → 1.307 |
| 0 | 6,304 / 6,304 | **0** | 0 / 0 | 4.016 → 0.751 → 0.662 | 0.750 → 0.664 |
| +6 | 406 / 406 | **0** | 0 / 0 | 1.009 → 0.0482 → 0.0432 | 0.0483 → 0.0432 |

The large NMSE drop from the *mixture* to the RGS estimate is mostly an
inappropriate attribution: the hard-demodulate/remodulate initialization has
already removed nearly all off-target-tone interference by construction. At
0 dB, for example, waveform NMSE is 4.016 for the raw mixture, 0.751 for the
non-learned hard initialization, then 0.662 after RGS. The model's incremental
gain is the last, much smaller change, not the full 4.016-to-0.662 difference.

For the three non-extreme SIRs, RGS lowers target-tone squared error by about
3.5% (-6), 11.5% (0), and 10.5% (+6) relative to the hard initialization,
yet none of the 100,800 paired bits crosses a decision boundary. Mean signed
margin on bits Direct got wrong moves from exactly `-1/√2 ≈ -0.707` at hard
initialization to about -0.647, -0.597, and -0.550 respectively: in the right
direction on average, but still on the wrong side of zero. At -24 dB RGS is
not merely ineffective at BER: active-tone error energy rises from 218,672
to 647,677, while non-target useful-tone error energy grows from ≈0 to
270,676. Its 90 bit flips net only 24 corrections and its NMSE is worse than
the hard initialization. Thus the earlier 5k-vs-Direct BER benefit at -24 dB
should not be interpreted as general waveform reconstruction quality.

## Interpretation and limit of the diagnosis

The **proximate, measured cause** of the BER plateau is that the present
inference trajectory leaves the Direct hard-bit decisions unchanged at the
examined -6/0/+6 dB audit parents. The frozen target prior was trained on
random QPSK target bit patterns; the hard initialization is itself a valid
target waveform regardless of whether those bits match the hidden truth.
It is therefore plausible that this starting point plus the current waveform
score update keeps the trajectory within its initial decision basin. The
interference prior/mixture constraint would need to provide enough evidence
to cross the correct axes; whether its score accuracy, the `ω=κ²` weighting,
the LR schedule, or the hard initialization is the dominant limiting factor
has **not** been isolated by this diagnostic. That is a hypothesis, not a
claim of a proven sign/scaling bug.

The train-statistics-based single-RE bit-MAP detector has lower BER than RGS
at most audit SIRs, so the received target REs do contain exploitable
information. Its advantage does not imply exact global optimality; it is an
estimated marginal detector. The next controlled investigation, if desired,
would isolate the RGS gradient terms and initialization on the audit split,
then evaluate any explicitly new receiver/inference variant there before
deciding whether another sealed formal test is warranted. The formal 5k test
and both monitoring automations remain paused.

The [machine-readable diagnostic summary](rgs_receiver_diagnosis_20260924.json)
contains pooled counts, spectral error energies, margins, and configuration
hash. Per-parent details and logs remain under ignored
`artifacts/rgs_ber_5000/receiver_diagnostic_20260924/`.
