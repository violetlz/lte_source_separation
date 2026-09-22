import numpy as np

from rfsep.waveforms.constellations import constellation_table, demap_symbols, map_bits
from rfsep.waveforms.ofdm import audit_cp, modulate_grid
from rfsep.waveforms.receiver import target_demodulate
from rfsep.waveforms.sources import generate_interference, generate_target


def test_constellation_exhaustive_and_independent_anchors():
    anchors = {
        "QPSK": ([0, 0], (1 + 1j) / np.sqrt(2)),
        "16QAM": ([0, 0, 0, 0], (1 + 1j) / np.sqrt(10)),
        "64QAM": ([0, 0, 0, 0, 0, 0], (3 + 3j) / np.sqrt(42)),
    }
    for modulation, (bits, anchor) in anchors.items():
        symbols, labels = constellation_table(modulation)
        assert np.array_equal(demap_symbols(symbols, modulation), labels)
        assert np.isclose(np.mean(np.abs(symbols) ** 2), 1.0, atol=1e-15)
        assert np.isclose(map_bits(np.array([bits], dtype=np.uint8), modulation)[0], anchor)


def test_resource_map_and_independent_single_tone(platform_config):
    cfg = platform_config
    assert cfg.target_subcarriers == tuple(range(-150, 0)) + tuple(range(1, 151))
    assert cfg.interference_subcarriers == tuple(range(-600, 0)) + tuple(range(1, 601))
    for tone in (-150, 150, -600, 600):
        tones = cfg.target_subcarriers if abs(tone) <= 150 else cfg.interference_subcarriers
        grid = np.zeros((14, len(tones)), dtype=np.complex128)
        grid[0, tones.index(tone)] = 1
        waveform = modulate_grid(grid, tones, cfg.fft_size, cfg.cp_lengths)
        indices = np.array([0, 1, 31, 2047])
        expected = np.exp(2j * np.pi * tone * indices / 2048) / np.sqrt(len(tones))
        cp = cfg.cp_lengths[0]
        assert np.max(np.abs(waveform[cp + indices] - expected)) < 1e-12


def test_target_roundtrip_cp_and_cp_damage_counterexample(platform_config):
    record, grid, bits = generate_target(platform_config, "audit", 5)
    assert record.waveform.shape == (30720,)
    cp = audit_cp(record.waveform, 2048, platform_config.cp_lengths)
    assert cp["cp_max_abs_error"] == 0
    recovered = target_demodulate(record.waveform, platform_config)
    assert np.linalg.norm(recovered - grid) / np.linalg.norm(grid) < 1e-10
    assert np.array_equal(demap_symbols(recovered, "QPSK").reshape(-1), bits)
    damaged = record.waveform.copy()
    damaged[:160] = 0
    assert audit_cp(damaged, 2048, platform_config.cp_lengths)["cp_max_abs_error"] > 0
    assert np.array_equal(target_demodulate(damaged, platform_config), recovered)


def test_interference_crop_boundaries_without_wrapping(platform_config):
    for tau in (0, 30719):
        record, long_waveform, maps = generate_interference(
            platform_config, "audit", 20 + tau, force_tau=tau, force_phi=0)
        expected = long_waveform[30720 + tau:61440 + tau]
        assert long_waveform.shape == (92160,)
        assert record.waveform.shape == (30720,)
        assert np.array_equal(record.waveform, expected)
        assert len(maps) == 3 and all(len(frame) == 4 for frame in maps)
