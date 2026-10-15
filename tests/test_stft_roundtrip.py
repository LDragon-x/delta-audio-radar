"""STFT/ISTFT round-trip (COLA) test for the frame-level model."""
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dsp.frame_model import stft, istft


def test_stft_istft_reconstruction():
    rs = np.random.RandomState(42)
    x = rs.randn(16000).astype(np.float32) * 0.2
    power, spec = stft(x)
    y = istft(spec)[:len(x)]
    err = np.linalg.norm(x - y) / np.linalg.norm(x)
    assert err < 0.05, "reconstruction error %f" % err
