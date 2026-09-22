# Migration notes

The legacy sibling repository was inspected at commit `a39a2fc2312fc6b5e4cdf0d1962a92f711ce627f`. Its LTE Oracle loader confirmed the historical `[14,2048]` CP-stripped grid and explicit mapping audits, but that path used fixed Oracle per-RE distributions and a different experiment boundary.

This project independently implements a continuous time-domain 5/20 MHz source distribution, valid asynchronous crop, fixed ensemble normalization, typed receiver/truth separation, and independent in-band calibration. It has no code import, checkpoint load, dataset read, or result dependency on the old repository. The initial pilot reused an already validated CUDA/PyTorch interpreter read-only while the slow Python 3.13 wheel download was avoided; no old environment package was modified. The delivered `.venv-train` is a project-local Python 3.10/CUDA 13 environment built from official wheels. No old result is presented as a new-platform result.

# Delivery B configuration identities

The physical canonical serialization and physical hash algorithm were not changed for Delivery B. Platform `lte_v1` therefore keeps hash `18b2a1bc425f217e36c9e67f9fd64f13701f45c83148ec0ccba42205f8115cb5`, and the existing calibration remains correctly bound to the same physical distribution.

Nested effective configuration is now returned by defensive copy, eliminating mutation-after-hash without changing canonical content. The new 0.5 ms view has its own hash and source-specific training settings have separate hashes. These identities must not be substituted for the physical hash or used to relabel a 0.5 ms model as a 1 ms prior.
