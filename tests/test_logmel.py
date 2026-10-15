"""Signal-property tests for the log-mel feature base (dsp/logmel.py)."""
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dsp.logmel import mel_bank, stft_power, hz_to_mel, mel_to_hz


def test_mel_bank_shape():
    sr, n_fft = 48000, 1024
    bank = mel_bank(sr, n_fft, n_mels=64)
    assert bank.shape == (64, n_fft // 2 + 1)
    assert bank.dtype == np.float32


def test_mel_bank_triangle_properties():
    # every filter: non-negative, single peak, peak value ~1
    bank = mel_bank(48000, 1024, n_mels=64)
    assert np.all(bank >= 0)
    # triangle apex never exceeds 1 and reaches ~1 somewhere
    peak = bank.max(axis=1)
    assert peak.max() <= 1.0 + 1e-5
    assert peak.max() > 0.99
    # every band carries energy (center not outside the FFT grid)
    assert np.all(bank.sum(axis=1) > 0)
    # band centers march upward: argmax non-decreasing across bands
    # (low bands share FFT bins, so not strictly increasing)
    argmax = bank.argmax(axis=1)
    assert np.all(np.diff(argmax) >= 0)


def test_hz_mel_roundtrip():
    f = np.array([100.0, 1000.0, 8000.0])
    assert np.allclose(mel_to_hz(hz_to_mel(f)), f, rtol=1e-6)


def test_stft_power_dims():
    sr = 48000
    x = np.random.RandomState(0).randn(sr).astype(np.float32) * 0.1
    pow_spec = stft_power(x, n_fft=512, hop=128)
    assert pow_spec.ndim == 2
    assert pow_spec.shape[1] == 257
    assert len(pow_spec) > 300
