"""
Unit and Integration Tests for Equilibrium Propagation Engine
==============================================================
"""

import numpy as np
import pytest
from src.equilibrium_propagation import MemristiveCrossbarNetwork, phase_preserving_saturation
from src.transducer import GaborWaveTransducer


def test_phase_preserving_saturation():
    """Verify that magnitude is bounded to [0, 1) and phase is strictly preserved."""
    z = np.array([3.0 + 4.0j, -5.0 + 0.0j, 0.0 + 2.0j, 0.0 + 0.0j])
    f_z = phase_preserving_saturation(z)

    # 1. Magnitude must be strictly less than 1.0 for non-zero inputs
    assert np.all(np.abs(f_z) < 1.0)
    # 2. Phase angles must match exactly for non-zero inputs
    nonzero = np.abs(z) > 1e-12
    orig_phase = np.angle(z[nonzero])
    sat_phase = np.angle(f_z[nonzero])
    assert np.allclose(orig_phase, sat_phase, atol=1e-10)


def test_free_phase_equilibrium_convergence():
    """Verify that state relaxes to a steady fixed point (dh/dt -> 0, dy/dt -> 0)."""
    np.random.seed(42)
    net = MemristiveCrossbarNetwork(dim_in=32, dim_hid=64, dim_out=32, dt=0.15)
    x = (np.random.normal(0, 1, 32) + 1j * np.random.normal(0, 1, 32)).astype(complex)
    x /= np.linalg.norm(x)

    # Relax for 30 steps and 40 steps
    h_30, y_30 = net.relax(x, steps=30)
    h_40, y_40 = net.relax(x, steps=40)

    # Difference between step 30 and step 40 is very small (exponential convergence)
    diff = float(np.max(np.abs(h_40 - h_30)))
    assert diff < 0.02, f"State failed to settle: diff = {diff}"


def test_noisy_wave_denoising_learning_convergence():
    """
    Workload Test: Train network to denoise distorted physical wave packets using Equilibrium Propagation.
    """
    np.random.seed(42)
    transducer = GaborWaveTransducer(n_grid=128, x_min=-10.0, x_max=10.0)
    clean_wave = transducer.encode("ai")

    dim = len(clean_wave)
    net = MemristiveCrossbarNetwork(
        dim_in=dim,
        dim_hid=128,
        dim_out=dim,
        eta=0.03,
        beta=0.3,
        dt=0.15
    )

    n_epochs = 25
    initial_loss = None
    final_loss = None

    for epoch in range(n_epochs):
        noise = (np.random.normal(0, 0.04, dim) + 1j * np.random.normal(0, 0.04, dim))
        noisy_input = clean_wave + noise
        norm = np.linalg.norm(noisy_input)
        if norm > 1e-12:
            noisy_input /= norm

        loss = net.train_step(noisy_input, clean_wave, free_steps=25, nudge_steps=12)
        if epoch == 0:
            initial_loss = loss
        final_loss = loss

    # Loss must have decreased through EqProp learning
    assert final_loss < initial_loss, f"Loss did not decrease: initial={initial_loss:.4e}, final={final_loss:.4e}"

    # Evaluate inference reconstruction on a fresh noisy test sample
    test_noise = (np.random.normal(0, 0.03, dim) + 1j * np.random.normal(0, 0.03, dim))
    test_input = clean_wave + test_noise
    test_input /= np.linalg.norm(test_input)

    denoised_wave = net.predict(test_input, steps=25)
    denoised_norm = np.linalg.norm(denoised_wave)
    if denoised_norm > 1e-12:
        denoised_wave /= denoised_norm

    fidelity = float(np.abs(np.vdot(clean_wave, denoised_wave)))
    assert fidelity > 0.65, f"Denoised wave fidelity too low: {fidelity:.4f}"
