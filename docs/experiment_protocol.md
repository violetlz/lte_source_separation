# Delivery A experiment protocol

1. Run deterministic tests and `rfsep audit`.
2. Estimate Q_b from 4,096 independent `calibration` records and verify Q_s on separate target records. Persist configuration and record-manifest identities.
3. A smoke run may use 16 examples, but it must reference an explicit calibration. A provisional calibration is never formal evidence.
4. The formal test uses 256 fixed test pairs at each SIR_RE in `[-12,-9,-6,-3,0,3,6,9,12]` dB. The same target/interference pair is reused across SIR; only κ and y change.
5. Score Clean target, Direct mixture, and Oracle subtraction audit. Clean and Oracle must have zero errors. Oracle is a link audit, never a method result.
6. Persist integer error/bit counts, every example status, actual in-band and total powers, fixed κ, runtime, configs, hashes, environment, and figures. Do not remove non-monotone or unfavorable Direct points.

The formal conclusion is limited to platform integrity and reproducibility. No learned prior, RGS, FM, LMMSE, SIC, or physical AWGN result is part of delivery A.
