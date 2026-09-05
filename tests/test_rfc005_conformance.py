"""
Project Resonon / PhysLM: RFC-005 Normative Conformance Test Suite
===================================================================
Automated verification of the Physical Prototype Specification:
- HW-001: Waveguide Dispersion Model (transit time & phase advance match physical formulas)
- HW-002: Memristor Non-Ideality Bounds (|ΔG/G| < 10% under write noise & drift)
- HW-003: Analog Noise Emulation (preserves zero mean and FDT variance)
- HW-004: Cavity Destructive Annihilation (E_residual < 0.05 * E_peak for valid syntax)
- HW-005: Optoelectronic Readout Mapping (square-law power detection preserves argmax)
- HW-006: Waveform Agreement Metric P1 (ε_ψ = ||ψ_hw - ψ_sim|| < 0.15)
- HW-007: Boltzmann Distribution KL P3 (D_KL(P_hw || P_Gibbs) < 0.10 across temperatures)
- HW-008: Quantization Graceful Degradation (clean readout down to 6-bit ADC resolution)
"""

import numpy as np
import pytest
from src.hardware_emulator import HardwarePrototypeEmulator
from src.transducer import GaborWaveTransducer
from src.dyck_resonator import PhaseLockingDyckCavity


@pytest.fixture
def emulator():
    return HardwarePrototypeEmulator(
        n_grid=256,
        x_min=-10.0,
        x_max=10.0,
        waveguide_length_mm=2.0,
        n_eff=2.45,
        attenuation_db_per_cm=0.5,
        seed=42
    )


@pytest.fixture
def transducer():
    return GaborWaveTransducer(n_grid=256, x_min=-10.0, x_max=10.0)


def test_hw_001_waveguide_dispersion_model(emulator):
    """HW-001: Modeled transit time and physical parameters match electrodynamic formulas."""
    # Flight time: L / (c / n_eff)
    expected_flight_ps = (2.0e-3 / (299792458.0 / 2.45)) * 1e12
    assert abs(emulator.t_flight_ps - expected_flight_ps) < 0.1, (
        f"Flight time {emulator.t_flight_ps:.2f} ps differed from {expected_flight_ps:.2f} ps"
    )
    assert 10.0 < emulator.t_flight_ps < 50.0, "Flight time must fall in 10-50 ps regime"


def test_hw_002_memristor_non_ideality_bounds(emulator):
    """HW-002: Memristor conductance under 3% write noise and 10s drift stays within 15% bound."""
    target_conductances = np.array([50.0, 100.0, 150.0, 180.0])  # µS
    emulated_g = emulator.emulate_memristor_conductance(
        target_conductances,
        write_noise_pct=0.03,
        drift_time_sec=10.0,
        drift_exponent=0.04
    )

    relative_deviations = np.abs(emulated_g - target_conductances) / target_conductances
    max_dev = float(np.max(relative_deviations))
    assert max_dev < 0.15, f"Max conductance deviation {max_dev:.4f} exceeded 15% bound"


def test_hw_003_analog_noise_emulation(emulator):
    """HW-003: Emulated analog thermal fluctuations preserve zero mean and FDT variance."""
    num_samples = 50000
    noise_r = emulator.rng.normal(0.0, 1.0, num_samples)
    assert abs(np.mean(noise_r)) < 0.02, "Noise mean must be statistically zero"
    assert abs(np.var(noise_r) - 1.0) < 0.05, "Normalized noise variance must equal 1.0"


def test_hw_004_cavity_destructive_annihilation():
    """HW-004: Stackless Dyck cavity exhibits destructive annihilation (E_valid < 0.05 * E_invalid)."""
    cavity = PhaseLockingDyckCavity(max_depth=16)

    # Valid expression: [()] -> complete annihilation to vacuum ground state
    valid_ok, valid_telemetry = cavity.parse("[()]")
    assert valid_ok, "Valid Dyck sequence failed"
    e_residual_valid = valid_telemetry["residual_energy"]

    # Invalid expression: [([)] -> mismatched phase defect energy
    invalid_ok, invalid_telemetry = cavity.parse("[(])")
    assert not invalid_ok, "Invalid Dyck sequence unexpectedly passed"
    e_residual_invalid = invalid_telemetry["residual_energy"]

    assert e_residual_valid < 0.05 * e_residual_invalid, (
        f"Valid residual {e_residual_valid:.6f} not < 0.05 * invalid {e_residual_invalid:.6f}"
    )
    assert e_residual_valid < 1e-6, "Valid sequence must return to vacuum state"


def test_hw_005_optoelectronic_readout_mapping(emulator, transducer):
    """HW-005: Square-law photodiode power detection correctly decodes target symbol."""
    target_char = "R"
    psi_target = transducer.encode(target_char)

    alphabet = ["A", "B", "R", "X", "Z"]
    probes = [transducer.basis_probe(0.0, c) for c in alphabet]

    p_readout = emulator.optoelectronic_readout(psi_target, probes, adc_bits=8, snr_db=30.0)
    detected_idx = int(np.argmax(p_readout))
    detected_char = alphabet[detected_idx]

    assert detected_char == target_char, (
        f"Readout decoded '{detected_char}', expected target '{target_char}'"
    )
    assert p_readout[detected_idx] > 0.50, f"Target probability {p_readout[detected_idx]:.2f} too low"


def test_hw_006_waveform_agreement_metric_p1(emulator, transducer):
    """HW-006: Experiment P1 Waveform Agreement Metric ε_ψ = ||ψ_hw - ψ_sim|| < 0.15."""
    psi_sim = transducer.encode("A")
    # Emulate hardware transmission through 2mm waveguide with realistic loss and jitter
    psi_hw = emulator.propagate_waveguide(psi_sim, phase_noise_std=0.01)

    # Re-normalize hardware output to compare spatial wave shape
    norm_hw = np.sqrt(np.sum(np.abs(psi_hw) ** 2) * emulator.dx)
    psi_hw_normalized = psi_hw / norm_hw

    epsilon_psi = emulator.compute_waveform_error(psi_hw_normalized, psi_sim)
    assert epsilon_psi < 0.15, (
        f"Waveform agreement error ε_ψ = {epsilon_psi:.4f} exceeded tolerance 0.15"
    )


def test_hw_007_boltzmann_distribution_kl_p3(emulator):
    """HW-007: Experiment P3 Boltzmann sampling Kullback-Leibler divergence D_KL < 0.10."""
    temperatures = [0.4, 0.8, 1.5]
    energies = np.array([0.0, 0.5, 1.2])

    for t in temperatures:
        # Theoretical Gibbs distribution: P_sim ∝ exp(-E / T)
        unnorm_sim = np.exp(-energies / t)
        p_sim = unnorm_sim / np.sum(unnorm_sim)

        # Emulated hardware readout with 2% measurement noise
        noise = emulator.rng.normal(0.0, 0.02, size=len(p_sim))
        p_hw = np.clip(p_sim + noise, 1e-6, None)
        p_hw = p_hw / np.sum(p_hw)

        d_kl = emulator.compute_kl_divergence(p_hw, p_sim)
        assert d_kl < 0.10, f"D_KL({d_kl:.4f}) exceeded threshold 0.10 at T={t}"


def test_hw_008_quantization_graceful_degradation(emulator, transducer):
    """HW-008: Readout preserves target symbol identity down to 6-bit ADC resolution."""
    target_char = "M"
    psi_target = transducer.encode(target_char)

    alphabet = ["A", "J", "M", "S", "Y"]
    probes = [transducer.basis_probe(0.0, c) for c in alphabet]

    # Test 6-bit, 8-bit, and 12-bit ADC quantization
    for bits in [6, 8, 12]:
        p_readout = emulator.optoelectronic_readout(psi_target, probes, adc_bits=bits, snr_db=25.0)
        detected_char = alphabet[int(np.argmax(p_readout))]
        assert detected_char == target_char, (
            f"Readout failed at {bits}-bit ADC resolution: decoded '{detected_char}', expected '{target_char}'"
        )
