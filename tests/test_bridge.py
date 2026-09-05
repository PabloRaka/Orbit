"""
Unit Tests for PyTorch & NumPy Tensor Bridge
=============================================
"""

import numpy as np
import pytest
from src.bridge import TensorStreamBridge, has_torch, numpy_to_torch, torch_to_numpy


def test_numpy_batch_streaming():
    """Verify pack and unpack of complex wave fields without torch."""
    bridge = TensorStreamBridge(use_torch=False)
    w1 = np.array([1.0 + 2.0j, 3.0 + 4.0j], dtype=complex)
    w2 = np.array([5.0 + 6.0j, 7.0 + 8.0j], dtype=complex)

    batch = bridge.pack_wave_batch([w1, w2])
    assert isinstance(batch, np.ndarray)
    assert batch.shape == (2, 2)

    unpacked = bridge.unpack_wave_batch(batch)
    assert len(unpacked) == 2
    assert np.allclose(unpacked[0], w1)
    assert np.allclose(unpacked[1], w2)


@pytest.mark.skipif(not has_torch(), reason="PyTorch not installed in environment")
def test_torch_interoperability_if_available():
    """Verify zero-copy PyTorch tensor round-trip if torch is available."""
    arr = np.array([1.0 + 1.0j, 2.0 - 2.0j], dtype=np.complex64)
    t = numpy_to_torch(arr)
    arr_back = torch_to_numpy(t)
    assert np.allclose(arr, arr_back)
