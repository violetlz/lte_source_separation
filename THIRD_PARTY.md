# Third-party and provenance notes

This repository's implementation was authored for this platform. It uses NumPy, PyYAML, Matplotlib, pytest, and SciPy under their respective upstream licenses.

The old `source_separation` repository at commit `a39a2fc2312fc6b5e4cdf0d1962a92f711ce627f` was inspected read-only for its LTE Oracle loader/audit conventions. No module is imported from the old repository and no source file was copied verbatim. The new mapping, OFDM, source, calibration, receiver, and evaluation code was written against this project's explicit contract.

The constellation/numerology cross-check refers to ETSI TS 136 211 V15.11.0. MATLAB reference generation uses the user's locally installed MathWorks MATLAB and LTE Toolbox; this repository does not redistribute those products or claim a license for them.
