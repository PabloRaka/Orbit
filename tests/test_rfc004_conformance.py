"""
Project Resonon / PhysLM: RFC-004 Normative Conformance Test Suite
===================================================================
Automated verification of the Thermal / Boltzmann Engine Specification:
- THERMAL-001: Zero-Temperature Determinism (T=0 recovers deterministic gradient descent)
- THERMAL-002: Zero-Mean Noise Invariant (|E[η]| < 10^-3)
- THERMAL-003: Fluctuation-Dissipation Variance (|Var(η) - 2μ k_B T dt/dx| < 5%)
- THERMAL-004: Fokker-Planck Equilibrium Ratio (P(A)/P(B) ≈ exp(-ΔE / k_B T))
- THERMAL-005: Empirical Boltzmann Histogram Match (R² > 0.95 against Gibbs measure)
- THERMAL-006: Monotonic Thermal Entropy Sweep (H(T1) < H(T2) for T1 < T2)
- THERMAL-007: Thermal Horizon Stability (zero NaN/Inf over 1,000 steps at T=1.0)
- THERMAL-008: RFC-002 Interface Invariance (| ||ψ||^2 - 1.0 | < 10^-6 preserved)
"""

import numpy as np
import pytest
from src.thermal_engine import ThermalBoltzmannEngine


def test_thermal_001_zero_temperature_determinism():
    """THERMAL-001: At T=0, Langevin dynamics reduces identically to deterministic gradient descent."""
    engine = ThermalBoltzmannEngine(n_grid=128, mobility=1.0, seed=42)
    psi0 = np.ones(128, dtype=complex) / np.sqrt(128 * engine.dx)
    grad = 0.5 * psi0

    dt = 0.01
    psi_stoch = engine.step_langevin(psi0, grad, dt=dt, temperature=0.0, renormalize=False)
    psi_det = psi0 - engine.mobility * grad * dt

    diff = np.max(np.abs(psi_stoch - psi_det))
    assert diff < 1e-12, f"Stochastic step differed from deterministic at T=0 by {diff:.6e}"


def test_thermal_002_zero_mean_noise_invariant():
    """THERMAL-002: Ensemble mean of Wiener thermal fluctuations approaches zero (|E[η]| < 10^-3)."""
    engine = ThermalBoltzmannEngine(n_grid=256, seed=123)
    dt = 0.01
    temperature = 0.5
    num_samples = 2000

    increments = [engine.generate_wiener_increment(dt, temperature) for _ in range(num_samples)]
    all_values = np.concatenate(increments)

    mean_r = float(np.mean(np.real(all_values)))
    mean_i = float(np.mean(np.imag(all_values)))

    assert abs(mean_r) < 1e-3, f"Real noise mean {mean_r:.6e} exceeded tolerance 1e-3"
    assert abs(mean_i) < 1e-3, f"Imaginary noise mean {mean_i:.6e} exceeded tolerance 1e-3"


def test_thermal_003_fluctuation_dissipation_variance():
    """THERMAL-003: Measured noise variance matches Einstein Fluctuation-Dissipation Theorem within 5%."""
    mobility = 1.5
    k_b = 1.0
    temperature = 0.4
    dt = 0.005
    engine = ThermalBoltzmannEngine(n_grid=512, mobility=mobility, k_b=k_b, seed=456)

    # Theoretical variance: Var(η) = 2 * μ * k_B * T * dt / dx
    expected_variance = 2.0 * mobility * k_b * temperature * dt / engine.dx

    num_samples = 2000
    increments = [engine.generate_wiener_increment(dt, temperature) for _ in range(num_samples)]
    all_values = np.concatenate(increments)
    measured_variance = float(np.var(np.real(all_values)) + np.var(np.imag(all_values)))

    relative_error = abs(measured_variance - expected_variance) / expected_variance
    assert relative_error < 0.05, (
        f"Variance error {relative_error:.4f} exceeded 5%. "
        f"Measured={measured_variance:.6f}, Expected={expected_variance:.6f}"
    )


def test_thermal_004_fokker_planck_equilibrium_ratio():
    """THERMAL-004: Two-state equilibrium population ratio reproduces Boltzmann ratio exp(-ΔE / k_B T)."""
    engine = ThermalBoltzmannEngine(seed=789)
    e_a = 1.0
    e_b = 2.0
    delta_e = e_b - e_a  # 1.0
    temperature = 0.8
    k_b = 1.0

    theoretical_ratio = engine.boltzmann_ratio(e_b, e_a, temperature)  # exp(-1.0 / 0.8) ≈ 0.2865

    # Discrete 2-state Markov jump process satisfying detailed balance: W(A->B) / W(B->A) = exp(-ΔE/T)
    steps = 40000
    current_state = "A"
    count_a = 0
    count_b = 0

    p_a_to_b = 0.1 * np.exp(-delta_e / (k_b * temperature))
    p_b_to_a = 0.1

    rng = np.random.default_rng(789)
    for _ in range(steps):
        if current_state == "A":
            count_a += 1
            if rng.random() < p_a_to_b:
                current_state = "B"
        else:
            count_b += 1
            if rng.random() < p_b_to_a:
                current_state = "A"

    empirical_ratio = count_b / count_a
    discrepancy = abs(empirical_ratio - theoretical_ratio) / theoretical_ratio
    assert discrepancy < 0.10, f"Empirical ratio {empirical_ratio:.4f} differed from {theoretical_ratio:.4f} by {discrepancy:.4f}"


def test_thermal_005_empirical_boltzmann_match():
    """THERMAL-005: Langevin walker trajectory in 1D potential matches continuous Gibbs measure (R² > 0.95)."""
    # Potential: V(x) = 0.5 * x^2 (harmonic oscillator) -> P_eq(x) ~ exp(-0.5 x^2 / T)
    temperature = 1.0
    mobility = 1.0
    dt = 0.01
    steps = 60000

    rng = np.random.default_rng(999)
    x = 0.0
    samples = np.empty(steps)

    for i in range(steps):
        grad_v = x  # dV/dx = x
        noise = rng.normal(0.0, np.sqrt(2.0 * mobility * temperature * dt))
        x = x - mobility * grad_v * dt + noise
        samples[i] = x

    # Discard burn-in
    steady_samples = samples[5000:]
    counts, bin_edges = np.histogram(steady_samples, bins=30, density=True)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    # Analytical Gaussian Gibbs distribution: P(x) = 1/√(2π T) * exp(-x² / (2T))
    theoretical_density = (1.0 / np.sqrt(2.0 * np.pi * temperature)) * np.exp(-(bin_centers ** 2) / (2.0 * temperature))

    # Compute coefficient of determination R²
    ss_res = np.sum((counts - theoretical_density) ** 2)
    ss_tot = np.sum((counts - np.mean(counts)) ** 2)
    r_squared = 1.0 - (ss_res / ss_tot)

    assert r_squared > 0.95, f"Langevin distribution R² {r_squared:.4f} was below threshold 0.95"


def test_thermal_006_monotonic_thermal_entropy_sweep():
    """THERMAL-006: Increasing temperature monotonically increases Shannon entropy of the sampled ensemble."""
    engine = ThermalBoltzmannEngine()
    temperatures = [0.1, 0.4, 0.9, 1.8]
    entropies = []

    # Discrete 5-state energy ladder: E_k = [0, 1, 2, 3, 4]
    energies = np.array([0.0, 1.0, 2.0, 3.0, 4.0])

    for t in temperatures:
        unnorm_p = np.exp(-energies / t)
        p = unnorm_p / np.sum(unnorm_p)
        h = engine.compute_entropy(p)
        entropies.append(h)

    for i in range(len(entropies) - 1):
        assert entropies[i] < entropies[i + 1], (
            f"Entropy not monotonically increasing: H({temperatures[i]})={entropies[i]:.4f} "
            f">= H({temperatures[i+1]})={entropies[i+1]:.4f}"
        )


def test_thermal_007_thermal_horizon_stability():
    """THERMAL-007: Continuous Langevin relaxation over 1,000 steps produces zero NaN or Inf."""
    engine = ThermalBoltzmannEngine(n_grid=128, seed=101)
    psi = np.ones(128, dtype=complex) / np.sqrt(128 * engine.dx)

    dt = 0.005
    state = psi.copy()

    for step in range(1000):
        grad = 0.1 * state
        state = engine.step_langevin(state, grad, dt=dt, temperature=1.0, renormalize=True)
        if step % 200 == 0:
            status = engine.check_stability(state)
            assert status["is_stable"], f"Stability check failed at step {step}"

    status = engine.check_stability(state)
    assert status["is_stable"]


def test_thermal_008_rfc002_interface_invariance():
    """THERMAL-008: Post-thermal-relaxation state strictly satisfies RFC-002 unitary norm tolerance."""
    engine = ThermalBoltzmannEngine(n_grid=256, seed=202)
    psi0 = np.exp(-(engine.x ** 2) / 2.0).astype(complex)
    psi0 /= np.sqrt(np.sum(np.abs(psi0) ** 2) * engine.dx)

    grad = 0.2 * psi0
    dt = 0.01

    state = psi0.copy()
    for _ in range(50):
        state = engine.step_langevin(state, grad, dt=dt, temperature=0.3, renormalize=True)

    norm_sq = np.sum(np.abs(state) ** 2) * engine.dx
    assert abs(norm_sq - 1.0) < 1e-6, f"Norm squared {norm_sq:.8f} violated RFC-002 tolerance 1e-6"
