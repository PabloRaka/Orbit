"""
Project Resonon / PhysLM: RFC-003 Normative Conformance Test Suite
===================================================================
Automated verification of the Wave Dynamics Engine Specification:
- WAVE-001: Free-wave propagation matches analytical dispersion (Δφ = β k² Δt)
- WAVE-002: Norm conservation at γ = 0 (|ΔN| < 10^-10)
- WAVE-003: Controlled dissipation decay at γ > 0 (||ψ(t)||^2 = e^(-2γt))
- WAVE-004: Numerical horizon stability (zero NaN/Inf over 500 steps)
- WAVE-005: Lattice refinement convergence (fine error < coarse error)
- WAVE-006: Split-step vs RK4 cross-solver agreement (dH < 10^-3)
- WAVE-007: Cavity eigenmode profile stability (stationary intensity)
- WAVE-008: Nonlinear amplitude remains bounded
"""

import numpy as np
import pytest
from src.wave_engine import WaveDynamicsEngine


def test_wave_001_analytical_dispersion_match():
    """WAVE-001: Phase advance of single plane wave matches analytical ω(k) = β k²."""
    n_grid = 256
    x_min, x_max = -10.0, 10.0
    beta = 0.5
    engine = WaveDynamicsEngine(n_grid=n_grid, x_min=x_min, x_max=x_max, beta=beta, gamma=0.0)

    # Pick integer mode index m = 3
    m = 3
    k0 = m * 2.0 * np.pi / engine.width
    psi0 = np.exp(1j * k0 * engine.x) / np.sqrt(engine.width)

    dt = 0.02
    psi_next = engine.step_split_step(psi0, dt=dt)

    # Analytical phase: exp(-i β k0² dt)
    expected_phase = -beta * (k0 ** 2) * dt
    analytical_psi = psi0 * np.exp(1j * expected_phase)

    error = np.sqrt(np.sum(np.abs(psi_next - analytical_psi) ** 2) * engine.dx)
    assert error < 1e-4, f"Analytical dispersion error {error:.6e} exceeded tolerance 1e-4"


def test_wave_002_norm_conservation_gamma_zero():
    """WAVE-002: Unitary probability norm is conserved to machine precision at γ = 0."""
    engine = WaveDynamicsEngine(n_grid=256, x_min=-10.0, x_max=10.0, beta=0.5, gamma=0.0)
    sigma = 0.8
    psi = np.exp(-(engine.x ** 2) / (2.0 * sigma ** 2)) * np.exp(1j * 3.0 * engine.x)
    psi /= np.sqrt(engine.compute_norm(psi))

    initial_norm = engine.compute_norm(psi)
    state = psi.copy()

    # Evolve 100 steps
    for _ in range(100):
        state = engine.step_split_step(state, dt=0.005)

    final_norm = engine.compute_norm(state)
    drift = abs(final_norm - initial_norm)
    assert drift < 1e-10, f"Norm drift {drift:.6e} exceeded 1e-10 under zero dissipation"


def test_wave_003_controlled_dissipation_decay():
    """WAVE-003: Under γ > 0, norm decays according to analytical law: ||ψ(t)||^2 = e^(-2γt)."""
    gamma = 0.5
    engine = WaveDynamicsEngine(n_grid=256, x_min=-10.0, x_max=10.0, beta=0.5, gamma=gamma)
    psi = np.exp(-(engine.x ** 2) / 2.0)
    psi /= np.sqrt(engine.compute_norm(psi))

    dt = 0.01
    steps = 50
    total_time = steps * dt  # 0.5 seconds

    state = psi.copy()
    for _ in range(steps):
        state = engine.step_split_step(state, dt=dt)

    final_norm = engine.compute_norm(state)
    expected_norm = np.exp(-2.0 * gamma * total_time)
    diff = abs(final_norm - expected_norm)
    assert diff < 1e-4, f"Dissipative decay norm {final_norm:.6f} differed from expected {expected_norm:.6f} by {diff:.6e}"


def test_wave_004_numerical_horizon_stability():
    """WAVE-004: Continuous propagation over 500 steps exhibits zero NaN/Inf or norm divergence."""
    engine = WaveDynamicsEngine(n_grid=256, x_min=-10.0, x_max=10.0, beta=0.5, g=0.05, gamma=0.0)
    psi = np.exp(-(engine.x ** 2) / 2.0) * np.exp(1j * 2.0 * engine.x)
    psi /= np.sqrt(engine.compute_norm(psi))

    state = psi.copy()
    for step in range(500):
        state = engine.step_split_step(state, dt=0.002)
        if step % 100 == 0:
            telemetry = engine.check_stability(state, initial_norm=1.0)
            assert telemetry["is_stable"], f"Stability check failed at step {step}"

    telemetry = engine.check_stability(state, initial_norm=1.0)
    assert telemetry["is_stable"]


def test_wave_005_lattice_refinement_convergence():
    """WAVE-005: Grid refinement (Δx -> 0) reduces error against analytical solution."""
    beta = 0.5
    dt = 0.01

    # Coarse lattice: N = 64
    engine_coarse = WaveDynamicsEngine(n_grid=64, x_min=-5.0, x_max=5.0, beta=beta)
    k_mode = 2.0 * 2.0 * np.pi / engine_coarse.width
    psi_coarse = np.exp(1j * k_mode * engine_coarse.x) / np.sqrt(engine_coarse.width)
    psi_coarse_next = engine_coarse.step_split_step(psi_coarse, dt=dt)
    expected_coarse = psi_coarse * np.exp(-1j * beta * (k_mode ** 2) * dt)
    err_coarse = np.sqrt(np.sum(np.abs(psi_coarse_next - expected_coarse) ** 2) * engine_coarse.dx)

    # Fine lattice: N = 256
    engine_fine = WaveDynamicsEngine(n_grid=256, x_min=-5.0, x_max=5.0, beta=beta)
    psi_fine = np.exp(1j * k_mode * engine_fine.x) / np.sqrt(engine_fine.width)
    psi_fine_next = engine_fine.step_split_step(psi_fine, dt=dt)
    expected_fine = psi_fine * np.exp(-1j * beta * (k_mode ** 2) * dt)
    err_fine = np.sqrt(np.sum(np.abs(psi_fine_next - expected_fine) ** 2) * engine_fine.dx)

    assert err_fine <= err_coarse + 1e-12, f"Fine error {err_fine:.6e} not <= coarse error {err_coarse:.6e}"


def test_wave_006_split_step_vs_rk4_agreement():
    """WAVE-006: Primary Split-Step Fourier agrees with reference RK4 solver within 10^-3."""
    engine = WaveDynamicsEngine(n_grid=256, x_min=-10.0, x_max=10.0, beta=0.5, g=0.02, gamma=0.0)
    psi0 = np.exp(-(engine.x ** 2) / 1.5) * np.exp(1j * 1.5 * engine.x)
    psi0 /= np.sqrt(engine.compute_norm(psi0))

    dt = 0.0005
    steps = 20

    # Integrate with Split-Step
    state_split = psi0.copy()
    for _ in range(steps):
        state_split = engine.step_split_step(state_split, dt=dt)

    # Integrate with RK4
    state_rk4 = psi0.copy()
    for _ in range(steps):
        state_rk4 = engine.step_rk4(state_rk4, dt=dt)

    discrepancy = np.sqrt(np.sum(np.abs(state_split - state_rk4) ** 2) * engine.dx)
    assert discrepancy < 1e-3, f"Split-Step vs RK4 discrepancy {discrepancy:.6e} exceeded tolerance 1e-3"


def test_wave_007_cavity_eigenmode_stability():
    """WAVE-007: Harmonic cavity eigenmode intensity profile |χ(x)|² remains stationary."""
    engine = WaveDynamicsEngine(n_grid=256, x_min=-5.0, x_max=5.0, beta=0.5, gamma=0.0)

    # Mode n = 1 standing wave in containment box
    w = engine.width
    chi1 = np.sqrt(2.0 / w) * np.sin(np.pi * (engine.x - engine.x_min) / w)
    chi1_c = chi1.astype(complex)

    state = chi1_c.copy()
    for _ in range(30):
        state = engine.step_split_step(state, dt=0.001)

    # In an eigenmode, intensity profile |ψ(x)|² should remain stationary
    intensity_init = np.abs(chi1_c) ** 2
    intensity_final = np.abs(state) ** 2
    profile_drift = np.max(np.abs(intensity_final - intensity_init))
    assert profile_drift < 0.05, f"Eigenmode intensity profile drifted by {profile_drift:.4f}"


def test_wave_008_nonlinear_amplitude_bound():
    """WAVE-008: Peak amplitude remains bounded in declared stable regime (no finite-time blowup)."""
    engine = WaveDynamicsEngine(n_grid=256, x_min=-10.0, x_max=10.0, beta=0.5, g=0.05, gamma=0.0)
    psi0 = np.exp(-(engine.x ** 2) / 1.0)
    psi0 /= np.sqrt(engine.compute_norm(psi0))

    initial_max = np.max(np.abs(psi0))
    state = psi0.copy()

    for _ in range(100):
        state = engine.step_split_step(state, dt=0.002)

    final_max = np.max(np.abs(state))
    ratio = final_max / initial_max
    assert ratio < 3.0, f"Nonlinear amplitude amplified excessively: ratio={ratio:.2f}"
