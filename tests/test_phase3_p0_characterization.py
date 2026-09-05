"""
Project Resonon / PhysLM: Phase III Gate P0 & P1 Conformance Test Suite
=======================================================================
Automated verification for Phase III: Physical Substrate Validation:
- Test 1: P0 Component parameter extraction (all 15 parameters in Θ_hw valid)
- Test 2: S_hw simulator initialization with extracted hardware parameters
- Test 3: Decoupled error metric decomposition (ε_ψ, ε_A, ε_φ)
- Test 4: Gate P1 Stop-Gate compliance (ε_ψ < 0.15, ε_φ < 0.20 rad)
- Test 5: Gate P1 Fault Trigger (intentionally degraded hardware trips Stop Gate)
- Test 6: Gate P3 MCMC diagnostics (autocorrelation time τ_corr and ESS)
"""

import numpy as np
import pytest
from src.hardware_calibrated_simulator import (
    ComponentCharacterizationHarness,
    HardwareCalibratedSimulator,
    HardwareParameters
)
from src.transducer import GaborWaveTransducer


def test_p0_component_parameter_extraction():
    """Gate P0: Extract all 15 parameters in Θ_hw and verify physical plausibility."""
    harness = ComponentCharacterizationHarness(seed=42)
    params = harness.extract_hardware_parameters()

    # Photonic domain bounds
    assert 0.2 < params.alpha_db_per_cm < 0.8, f"Alpha {params.alpha_db_per_cm} out of physical range"
    assert -2.0 < params.beta2_ps2_per_m < -0.5, f"Beta2 {params.beta2_ps2_per_m} out of range"
    assert 2.40 < params.n_eff < 2.50, f"n_eff {params.n_eff} out of range"
    assert 0.005 < params.phase_noise_std < 0.03, f"Phase jitter {params.phase_noise_std} out of range"

    # Memristor domain bounds
    assert 5.0 < params.g_min_us < 20.0, f"G_min {params.g_min_us} out of range"
    assert 150.0 < params.g_max_us < 250.0, f"G_max {params.g_max_us} out of range"
    assert 0.01 < params.write_noise_pct < 0.06, f"Write noise {params.write_noise_pct} out of range"
    assert 0.02 < params.drift_exponent_nu < 0.08, f"Drift nu {params.drift_exponent_nu} out of range"

    # Thermal & readout domain bounds
    assert abs(params.noise_mean_offset) < 0.01, f"Noise offset {params.noise_mean_offset} too large"
    assert 6.5 < params.effective_adc_bits < 10.0, f"ENOB {params.effective_adc_bits} out of range"
    assert 25.0 < params.snr_db < 45.0, f"SNR {params.snr_db} out of range"


def test_p0_hardware_calibrated_simulator_initialization():
    """Gate P0: Calibrated simulator S_hw initializes with valid physical constants."""
    params = HardwareParameters()
    sim = HardwareCalibratedSimulator(params=params, n_grid=256, seed=42)

    assert sim.v_g > 1.0e8, "Group velocity must exceed 10^8 m/s"
    assert 10.0 < sim.t_flight_ps < 30.0, f"Transit time {sim.t_flight_ps} ps outside expected range"


def test_p1_decoupled_error_decomposition():
    """Gate P1: Decomposition of waveform discrepancy into amplitude and phase components."""
    dx = 0.05
    n_grid = 256
    x = np.linspace(-5.0, 5.0, n_grid)
    psi_ref = np.exp(-(x ** 2) / 2.0).astype(complex)
    psi_ref /= np.sqrt(np.sum(np.abs(psi_ref) ** 2) * dx)

    # 1. Pure phase rotation: |ψ_phase| == |ψ_ref|
    phase_shift = 0.10  # radians
    psi_phase = psi_ref * np.exp(1j * phase_shift)
    errors_phase = HardwareCalibratedSimulator.compute_decoupled_errors(psi_phase, psi_ref, dx)

    assert errors_phase["epsilon_A"] < 1e-10, "Pure phase shift must have zero amplitude error"
    assert abs(errors_phase["epsilon_phi"] - phase_shift) < 1e-4, "Phase error must equal phase shift"

    # 2. Pure amplitude attenuation
    attenuation_factor = 0.90
    psi_atten = psi_ref * attenuation_factor
    errors_atten = HardwareCalibratedSimulator.compute_decoupled_errors(psi_atten, psi_ref, dx)

    assert errors_atten["epsilon_phi"] < 1e-10, "Pure attenuation must have zero phase error"
    assert abs(errors_atten["epsilon_A"] - 0.10) < 1e-4, "Amplitude error must equal 1 - attenuation"


def test_p1_stop_gate_pass_criterion():
    """Gate P1: Calibrated hardware propagation satisfies Stop Gate criteria (ε_ψ < 0.15, ε_φ < 0.20 rad)."""
    params = HardwareParameters()
    sim = HardwareCalibratedSimulator(params=params, n_grid=256, seed=42)
    transducer = GaborWaveTransducer(n_grid=256, x_min=-10.0, x_max=10.0)

    psi_sim = transducer.encode("P")
    psi_hw = sim.simulate_waveguide_step(psi_sim)

    # Re-normalize hardware output to isolate spatial mode shape and phase
    norm_hw = np.sqrt(np.sum(np.abs(psi_hw) ** 2) * sim.dx)
    psi_hw_norm = psi_hw / norm_hw

    errors = HardwareCalibratedSimulator.compute_decoupled_errors(psi_hw_norm, psi_sim, sim.dx)

    assert errors["epsilon_psi"] < 0.15, f"Total error {errors['epsilon_psi']:.4f} exceeded Stop Gate 0.15"
    assert errors["epsilon_phi"] < 0.20, f"Phase error {errors['epsilon_phi']:.4f} exceeded Stop Gate 0.20 rad"


def test_p1_stop_gate_fault_trigger():
    """Gate P1: Severe hardware phase jitter correctly trips the Stop Gate."""
    degraded_params = HardwareParameters(phase_noise_std=0.45)  # Severe laser phase noise
    sim_degraded = HardwareCalibratedSimulator(params=degraded_params, n_grid=256, seed=99)
    transducer = GaborWaveTransducer(n_grid=256, x_min=-10.0, x_max=10.0)

    psi_sim = transducer.encode("X")
    psi_hw = sim_degraded.simulate_waveguide_step(psi_sim)
    norm_hw = np.sqrt(np.sum(np.abs(psi_hw) ** 2) * sim_degraded.dx)
    psi_hw_norm = psi_hw / norm_hw

    errors = HardwareCalibratedSimulator.compute_decoupled_errors(psi_hw_norm, psi_sim, sim_degraded.dx)
    # The Stop Gate must detect this and fail
    is_halted = (errors["epsilon_psi"] >= 0.15) or (errors["epsilon_phi"] >= 0.20)
    assert is_halted, f"Stop Gate failed to trip on degraded hardware: ε_ψ={errors['epsilon_psi']:.4f}, ε_φ={errors['epsilon_phi']:.4f}"


def test_p3_mcmc_diagnostics_autocorr_and_ess():
    """Gate P3: Verify integrated autocorrelation time τ_corr and Effective Sample Size (ESS)."""
    # 1. Uncorrelated white noise: τ_corr ≈ 1.0, ESS ≈ N / 2
    n_samples = 4000
    rng = np.random.default_rng(123)
    uncorr_samples = rng.normal(0.0, 1.0, n_samples)
    diag_uncorr = HardwareCalibratedSimulator.compute_mcmc_diagnostics(uncorr_samples)

    assert abs(diag_uncorr["tau_corr"] - 1.0) < 0.5, f"Uncorrelated τ_corr {diag_uncorr['tau_corr']} not near 1.0"
    assert diag_uncorr["ess"] > 1000.0, f"Uncorrelated ESS {diag_uncorr['ess']} too small"

    # 2. Highly correlated AR(1) process: x_t = 0.85 * x_{t-1} + noise
    corr_samples = np.empty(n_samples)
    x = 0.0
    for i in range(n_samples):
        x = 0.85 * x + rng.normal(0.0, 0.5)
        corr_samples[i] = x

    diag_corr = HardwareCalibratedSimulator.compute_mcmc_diagnostics(corr_samples)
    assert diag_corr["tau_corr"] > 3.0, f"Correlated τ_corr {diag_corr['tau_corr']} should be > 3.0"
    assert diag_corr["ess"] < diag_uncorr["ess"], "Correlated ESS must be less than uncorrelated ESS"
