# Prior dataset v1 validation

Dataset ID: `lte-prior-v1-parent32000-n200000-val1024`.

## Materialized records

| Source/split | Independent parents | Rows/windows | Parent IDs | Waveform bytes | Generation time |
|---|---:|---:|---|---:|---:|
| target/train | 200,000 | 200,000 | 32000..231999 | 24,576,050,048 | 118.0 s |
| interference/train | 200,000 | 200,000 | 32000..231999 | 24,576,050,048 | 169.1 s |
| target/validation | 1,024 | 2,048 | 0..1023 | 251,658,752 | 3.1 s |
| interference/validation | 1,024 | 2,048 | 0..1023 | 251,658,752 | 5.8 s |

The on-disk directory is about 47 GiB including NPY headers and audit sidecars. Each row is
float32 `[2,15360]` in the existing `sqrt(2)[Re,Im]` coordinate system. Training uses one
deterministic slot per parent; validation stores both slots. The old 64-parent online
validation remains available as a legacy protocol and was not overwritten.

## Full validation

Every shard passed full shape, dtype, finite-value, byte-count and SHA256 verification.
The validator regenerated fixed rows spanning first/middle/last shards and required exact
array equality with the original generator. Parent counts/ID ranges were checked for
duplicates and omissions before each manifest state changed from `generated` to
`validated`; the training Dataset refuses any other state.

The full-row complex mean powers were 1.00000874 (target train), 0.99999216
(interference train), 0.99995735 (target validation), and 1.00005273 (interference
validation). Regenerated target checks had exact CP equality and zero errors over 33,600
bits in each split sample. Window FFT PSD diagnostics were finite. For asynchronous
interference, the validator locates only complete source symbols using the recorded crop
offset; 48 complete-symbol CP pairs in the fixed validation regeneration sample had zero
maximum error.

Target train slot counts were 100,001/99,999. Interference train slot counts were
99,618/100,382. Interference tau covered 0..30719 and every one of 32 tau and phi bins was
populated. Across three generated frames and four blocks, modulation counts were QPSK
798,923, 16QAM 800,060 and 64QAM 801,017.

For the actually visible window portions (not merely generated frames), all 81 four-block
modulation maps occurred; counts ranged from 3,524 to 3,939 over 299,943 visible-frame
observations. 99,943/200,000 windows crossed an interference subframe boundary
(fraction 0.499715); 100,057 did not.

Train, validation, RGS-dev and formal-test use separate split/ID namespaces. No audit
metadata is returned by the training Dataset. The 3.2M/4.8M eventual sample exposures are
repeated epoch visits over 200,000 unique training parents, not millions of independent
records.
