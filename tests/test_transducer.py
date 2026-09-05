"""
Unit and Integration Tests for Tokenless Gabor Wave Transducer
==============================================================
"""

import numpy as np
import pytest
from src.transducer import GaborWaveTransducer
from src.baseline_phase0 import ContinuousWaveEngine


def test_unitary_normalization():
    """Verify that any encoded text produces a strictly normalized Hilbert state."""
    transducer = GaborWaveTransducer(n_grid=1024)
    for sample in ["A", "cat", "Hello Orbit!", "12345"]:
        psi = transducer.encode(sample)
        norm = np.sum(np.abs(psi) ** 2) * transducer.dx
        assert np.isclose(norm, 1.0, atol=1e-6), f"Norm failed for '{sample}': {norm}"


def test_round_trip_reconstruction():
    """Verify exact character recovery from wave field."""
    transducer = GaborWaveTransducer(n_grid=1024)
    test_cases = [
        "cat",
        "orbit",
        "quantum",
        "fisika"
    ]
    for text in test_cases:
        psi = transducer.encode(text)
        recovered = transducer.decode(psi, expected_length=len(text))
        assert recovered == text, f"Failed round-trip: expected '{text}', got '{recovered}'"


def test_noise_robustness():
    """Verify that projective quantum measurement tolerates analog thermal noise."""
    transducer = GaborWaveTransducer(n_grid=1024)
    original_text = "orbit"
    psi = transducer.encode(original_text)

    # Add Gaussian thermal noise (sigma = 0.02)
    np.random.seed(42)
    noise = (np.random.normal(0, 0.02, len(psi)) + 1j * np.random.normal(0, 0.02, len(psi))) * np.sqrt(transducer.dx)
    psi_noisy = psi + noise
    psi_noisy /= np.sqrt(np.sum(np.abs(psi_noisy) ** 2) * transducer.dx)

    recovered = transducer.decode(psi_noisy, expected_length=len(original_text))
    assert recovered == original_text, f"Noise test failed: expected '{original_text}', got '{recovered}'"


def test_transducer_coupled_with_wave_engine():
    """
    Integration Test: Encode text -> Propagate wave through physical Hamiltonian -> Verify unitary norm.
    """
    transducer = GaborWaveTransducer(n_grid=1024, x_min=-20.0, x_max=20.0)
    psi_text = transducer.encode("ai")

    engine = ContinuousWaveEngine(n_grid=1024, x_min=-20.0, x_max=20.0, g=0.0)
    v_flat = np.zeros(1024)

    # Propagate for 100 physical time steps (dt = 0.002)
    psi = psi_text.copy()
    for _ in range(100):
        psi = engine.step_unitary_split_operator(psi, v_flat, dt=0.002)

    norm_final = engine.compute_norm(psi)
    assert np.isclose(norm_final, 1.0, atol=1e-10), f"Wave propagation leaked norm: {norm_final}"
