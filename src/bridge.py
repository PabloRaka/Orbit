"""
Project Resonon / PhysLM: PyTorch & NumPy Zero-Copy Interoperability Bridge
===========================================================================
Specification Reference: docs/backbone/03_SOFTWARE_SIMULATION_SPECIFICATION.md

Enables high-throughput, zero-copy data exchange between deep learning
ecosystem tensors (PyTorch / DLPack) and physical continuous complex wave fields.
"""

import numpy as np
from typing import Any, Union


def has_torch() -> bool:
    """Checks if PyTorch is available in current environment."""
    try:
        import torch
        return True
    except ImportError:
        return False


def numpy_to_torch(array: np.ndarray) -> Any:
    """
    Converts a NumPy complex/real array to a PyTorch tensor via zero-copy memory sharing.
    """
    import torch
    return torch.from_numpy(array)


def torch_to_numpy(tensor: Any) -> np.ndarray:
    """
    Converts a PyTorch tensor to a NumPy array without data duplication.
    """
    if hasattr(tensor, "detach"):
        tensor = tensor.detach()
    if hasattr(tensor, "cpu"):
        tensor = tensor.cpu()
    if hasattr(tensor, "numpy"):
        return tensor.numpy()
    return np.asarray(tensor)


class TensorStreamBridge:
    """
    Manages continuous streaming batches between external data loaders
    and the PhysLM continuous wave simulation.
    """
    def __init__(self, use_torch: bool = None):
        self.use_torch = has_torch() if use_torch is None else use_torch

    def pack_wave_batch(self, waves: list[np.ndarray]) -> Any:
        """Packs a list of 1D wave states into a unified 2D batch tensor."""
        batch_np = np.stack(waves, axis=0)
        if self.use_torch:
            return numpy_to_torch(batch_np)
        return batch_np

    def unpack_wave_batch(self, batch_tensor: Any) -> list[np.ndarray]:
        """Unpacks a 2D batch tensor into a list of 1D wave states."""
        batch_np = torch_to_numpy(batch_tensor) if self.use_torch else batch_tensor
        return [batch_np[i] for i in range(batch_np.shape[0])]
