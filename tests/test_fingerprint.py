"""Voiceprint fingerprint properties (dsp/guntype.py)."""
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dsp.guntype import fingerprint, cosine_dist


def _burst(freq, sr=48000, n=2048):
    t = np.arange(n) / sr
    x = np.sin(2 * np.pi * freq * t)
    x *= np.hanning(n)
    return x.astype(np.float32)


def test_identical_windows_zero_distance():
    w = _burst(440)
    a = fingerprint(w)
    assert cosine_dist(a, a) < 1e-6


def test_different_bursts_positive_distance():
    a = fingerprint(_burst(440))
    b = fingerprint(_burst(2000))
    assert cosine_dist(a, b) > 0.1
