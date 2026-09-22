# Data contract

`SourceRecord` belongs to generation/data code and carries a normalized time-domain source plus parent identity and audit metadata. `Observation` contains only an opaque example ID and received time-domain IQ. `ReceiverInfo` contains the version/hash, calibration identity, κ, nominal in-band SIR, and the zero-noise declaration. `GroundTruth` contains clean time-domain sources, target bits/grid, interference allocation, tau/phi, and parent identities. `ReceiverOutput` contains recovered target REs/bits and optionally an estimated target waveform.

Manifests derive local RNGs with SHA-256 over `(scheme, root seed, purpose, split, record ID)`. Python's process-randomized `hash()` and global NumPy state are not used. Parent records are assigned to one of `train`, `validation`, `test`, `calibration`, or `audit` before a crop is generated. All crops of a parent retain that split.

Observation and truth are never serialized as one catch-all metadata dictionary. A detector signature may accept `Observation`, `ReceiverInfo`, public config/prior/budget, and its own RNG; it must not accept `GroundTruth` or a manifest.

## Prior-training view

The physical parent remains one 1 ms, 30,720-complex-sample record. `prior_view_v1` exposes a 0.5 ms, 15,360-complex-sample training window starting only at 0 or 15,360. The deterministic slot choice is derived from `(root seed, source, split, parent ID)`; both slots keep the parent's split. Validation evaluates both slots of 64 fixed parents. For interference, the platform first constructs its asynchronous, phase-rotated 1 ms receiver crop and only then applies the half-record view.

Target and interference datasets call their respective source generators directly. They never call the mixture builder and never generate the other source. Training cursors count only samples committed by successful optimizer updates; DataLoader prefetch is therefore not part of resume identity.

Network coordinates are unambiguously channels-first:

```text
complex [L]   <-> float32 [2,L]
complex [B,L] <-> float32 [B,2,L]
R(x) = sqrt(2) * stack(real(x), imag(x), channel_axis)
```

There is no per-record or per-window normalization. The inverse returns complex64. For a unit-average-power complex ensemble, the mean square over both real channels is approximately one.

`WaveformSpec` is the receiver-visible geometry and platform identity. It intentionally excludes the root seed and generation truth. The receiver accepts only `WaveformSpec`, `ReceiverInfo`, and the observation; tau, phi, modulation maps, bits, clean sources, SIR truth, and full generation configuration remain outside the model call.

The platform physical-distribution hash, prior-view hash, and source-specific training-config hash are separate. Effective platform and training configuration access returns defensive copies, preventing later nested mutation from silently invalidating a cached hash.
