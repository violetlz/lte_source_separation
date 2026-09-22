# Platform design v1

## Purpose and status

This is the frozen design for delivery A of the LTE-inspired 5/20 MHz asynchronous OFDM source-separation benchmark. The closed loop runs from random bits to a time-domain waveform and back to target bits. It does not claim that a two-source separation method exists yet.

## Physical model

Both sources use `Fs=30.72 MHz`, `N=2048`, 15 kHz spacing, 14 OFDM symbols, normal CP `[160,144,144,144,144,144,144]` repeated twice, and a 30,720-sample observation. The target occupies signed tones `[-150..-1,1..150]` with QPSK. The interference occupies `[-600..-1,1..600]`; four ordered 300-tone blocks independently choose QPSK, 16QAM, or 64QAM for each subframe.

Each source generates three continuous subframes. The target uses the aligned middle frame. Interference uses one integer `tau∈[0,L-1]`, one `phi∈[0,2π)`, and the valid crop `exp(j phi)b_long[L+tau:2L+tau]`. There is no circular wrapping. The v1 channel is scalar one, windowing is off, and physical noise is exactly zero.

Natural FFT bin indexing is `k mod 2048`; `fftshift` is display-only. Grid order is `[symbol, active_subcarrier]`, and bit order is `[symbol, active_subcarrier, bit_in_symbol]` in C order.

## Scale and receiver

FFT/IFFT uses `norm="ortho"`. A K-tone source has fixed scale `c_K=sqrt(N/K)`. No record is normalized by its measured RMS. The one target receiver removes the known target CP, FFTs each useful symbol, extracts the 300 target tones, divides by `c_300`, and applies the one QPSK decision rule.

In-band calibration estimates `Q_b=E||Db||²/4200` from an independent calibration split. For requested `gamma=SIR_RE`, `kappa=sqrt(1/(Q_b*10^(gamma/10)))` and `y=s+kappa*b`. The same κ is used for every example at one condition.

## Information boundary

The detector sees only `Observation(y, example_id)` and declared `ReceiverInfo`. Clean sources, bits, tau, phi, modulation maps, parent records, and per-example powers remain in `GroundTruth` and are available only to generation, audit, and scoring.

The project architecture supplied with the task is preserved in `docs/project_architecture.md`. Delivery A deliberately omits empty model/training/prior/solver stubs.
