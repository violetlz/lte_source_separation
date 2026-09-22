# Data contract

`SourceRecord` belongs to generation/data code and carries a normalized time-domain source plus parent identity and audit metadata. `Observation` contains only an opaque example ID and received time-domain IQ. `ReceiverInfo` contains the version/hash, calibration identity, κ, nominal in-band SIR, and the zero-noise declaration. `GroundTruth` contains clean time-domain sources, target bits/grid, interference allocation, tau/phi, and parent identities. `ReceiverOutput` contains recovered target REs/bits and optionally an estimated target waveform.

Manifests derive local RNGs with SHA-256 over `(scheme, root seed, purpose, split, record ID)`. Python's process-randomized `hash()` and global NumPy state are not used. Parent records are assigned to one of `train`, `validation`, `test`, `calibration`, or `audit` before a crop is generated. All crops of a parent retain that split.

Observation and truth are never serialized as one catch-all metadata dictionary. A detector signature may accept `Observation`, `ReceiverInfo`, public config/prior/budget, and its own RNG; it must not accept `GroundTruth` or a manifest.
