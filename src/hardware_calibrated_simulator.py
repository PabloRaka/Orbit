"""
Project Resonon / PhysLM: Phase III Hardware Characterization & Calibrated Simulator
====================================================================================
Implements Gate P0 and the Hardware-in-the-Loop (HIL) Simulator S_hw:
1. Component Characterization: Extracts empirical parameters Θ_hw across all four physical domains.
2. Calibrated Simulation (S_hw): Integrates wave dynamics under empirical Θ_hw.
3. Decoupled Error Metrics: Computes amplitude error ε_A, phase error ε_φ, and total error ε_ψ.
4. MCMC Diagnostics: Computes integrated autocorrelation time τ_corr and Effective Sample Size (ESS).
"""

import numpy as np
from typing import Dict, Any, Tuple, Optional, List
from dataclasses import dataclass, field


@dataclass
class HardwareParameters:
    """Empirical parameter set Θ_hw extracted from physical component characterization."""
    # Photonic Waveguide
    alpha_db_per_cm: float = 0.48
    beta2_ps2_per_m: float = -1.18
    n_eff: float = 2.452
    length_mm: float = 2.0
    laser_linewidth_mhz: float = 0.1
    phase_noise_std: float = 0.012

    # Memristive Crossbar
    g_min_us: float = 12.5
    g_max_us: float = 195.0
    write_noise_pct: float = 0.032
    drift_exponent_nu: float = 0.041
    retention_time_sec: float = 1000.0

    # Thermal Noise Source
    noise_mean_offset: float = 0.0001
    effective_temp_ratio: float = 0.25
    thermal_psd_flatness_db: float = 0.8

    # Optoelectronic Readout
    dark_current_na: float = 1.2
    dynamic_range_db: float = 42.0
    effective_adc_bits: float = 7.8
    snr_db: float = 34.0


class ComponentCharacterizationHarness:
    """Executes Gate P0: Physical Characterization of Components."""

    def __init__(self, seed: Optional[int] = None):
        self.rng = np.random.default_rng(seed)

    def extract_hardware_parameters(self) -> HardwareParameters:
        """
        Extracts empirical parameter set Θ_hw from hardware testbed measurements.
        Simulates laboratory instrument readouts (OSA, parameter analyzer, oscilloscope).
        """
        # Photonic waveguide measurement (cut-back & delay)
        measured_alpha = float(0.45 + self.rng.normal(0.0, 0.02))
        measured_beta2 = float(-1.20 + self.rng.normal(0.0, 0.03))
        measured_neff = float(2.450 + self.rng.normal(0.0, 0.005))
        phase_jitter = float(0.012 + self.rng.normal(0.0, 0.001))

        # Memristor array measurement (I-V and drift testing)
        g_min = float(12.0 + self.rng.normal(0.0, 0.5))
        g_max = float(196.0 + self.rng.normal(0.0, 2.0))
        write_noise = float(0.030 + self.rng.normal(0.0, 0.002))
        drift_nu = float(0.040 + self.rng.normal(0.0, 0.002))

        # Thermal circuit measurement (RF spectrum analyzer)
        noise_mean = float(self.rng.normal(0.0, 0.0002))

        # Optoelectronic detector characterization
        enob = float(7.8 + self.rng.normal(0.0, 0.1))
        snr = float(34.5 + self.rng.normal(0.0, 0.5))

        return HardwareParameters(
            alpha_db_per_cm=measured_alpha,
            beta2_ps2_per_m=measured_beta2,
            n_eff=measured_neff,
            length_mm=2.0,
            phase_noise_std=phase_jitter,
            g_min_us=g_min,
            g_max_us=g_max,
            write_noise_pct=write_noise,
            drift_exponent_nu=drift_nu,
            noise_mean_offset=noise_mean,
            effective_adc_bits=enob,
            snr_db=snr
        )


class HardwareCalibratedSimulator:
    """
    Hardware-in-the-Loop Simulator S_hw:
    Simulates continuous wave and attractor evolution parameterized by empirical Θ_hw.
    """

    def __init__(
        self,
        params: HardwareParameters,
        n_grid: int = 256,
        x_min: float = -10.0,
        x_max: float = 10.0,
        seed: Optional[int] = None
    ):
        self.params = params
        self.n_grid = n_grid
        self.x_min = x_min
        self.x_max = x_max
        self.x = np.linspace(x_min, x_max, n_grid, endpoint=False)
        self.dx = float(self.x[1] - self.x[0])
        self.width = float(x_max - x_min)

        self.c_light = 299792458.0
        self.v_g = self.c_light / params.n_eff
        self.t_flight_ps = float((params.length_mm * 1e-3 / self.v_g) * 1e12)

        # Attenuation coefficient α in [1/m]
        alpha_db_per_m = params.alpha_db_per_cm * 100.0
        self.alpha_m = float(alpha_db_per_m * np.log(10.0) / 10.0)

        # GVD in s^2 / m
        self.beta2_s2_per_m = float(params.beta2_ps2_per_m * 1e-24)

        self.rng = np.random.default_rng(seed)

    def simulate_waveguide_step(self, psi_in: np.ndarray) -> np.ndarray:
        """Propagates wavefield through calibrated physical waveguide."""
        length_m = self.params.length_mm * 1e-3
        attenuation = np.exp(-0.5 * self.alpha_m * length_m)

        # Spatial and temporal phase jitter across waveguide transverse profile
        phase_jitter = self.rng.normal(0.0, self.params.phase_noise_std, size=self.n_grid)

        # Dispersion
        k = 2.0 * np.pi * np.fft.fftfreq(self.n_grid, d=self.dx)
        dispersion_phase = -0.5 * self.beta2_s2_per_m * (self.v_g ** 2) * (k ** 2) * length_m

        psi_k = np.fft.fft(psi_in)
        psi_k = psi_k * np.exp(1j * dispersion_phase)
        psi_out = np.fft.ifft(psi_k) * attenuation * np.exp(1j * phase_jitter)

        return psi_out

    @staticmethod
    def compute_decoupled_errors(
        psi_hw: np.ndarray,
        psi_sim: np.ndarray,
        dx: float
    ) -> Dict[str, float]:
        """
        Decomposes total waveform discrepancy ε_ψ into amplitude error ε_A and phase error ε_φ:
            ε_ψ = ||ψ_hw - ψ_sim|| / ||ψ_sim||
            ε_A = || |ψ_hw| - |ψ_sim| || / || |ψ_sim| ||
            ε_φ = √[ ∫ |angle(ψ_hw) - angle(ψ_sim)|² (|ψ_sim|² / ||ψ_sim||²) dx ]
        """
        norm_sim = np.sqrt(np.sum(np.abs(psi_sim) ** 2) * dx)
        diff_total = psi_hw - psi_sim
        eps_psi = float(np.sqrt(np.sum(np.abs(diff_total) ** 2) * dx) / max(1e-12, norm_sim))

        amp_hw = np.abs(psi_hw)
        amp_sim = np.abs(psi_sim)
        diff_amp = amp_hw - amp_sim
        eps_a = float(np.sqrt(np.sum(diff_amp ** 2) * dx) / max(1e-12, norm_sim))

        # Phase error weighted by localized wave intensity
        phase_hw = np.angle(psi_hw)
        phase_sim = np.angle(psi_sim)
        # Phase difference wrapped to [-pi, pi]
        phase_diff = np.angle(np.exp(1j * (phase_hw - phase_sim)))
        weight = (amp_sim ** 2) / max(1e-12, norm_sim ** 2)
        eps_phi = float(np.sqrt(np.sum((phase_diff ** 2) * weight * dx)))

        return {
            "epsilon_psi": eps_psi,
            "epsilon_A": eps_a,
            "epsilon_phi": eps_phi
        }

    @staticmethod
    def compute_mcmc_diagnostics(samples: np.ndarray, max_lag: int = 100) -> Dict[str, float]:
        """
        Computes integrated autocorrelation time τ_corr and Effective Sample Size (ESS):
            τ_corr = 1 + 2 Σ_{k=1}^W ρ(k)
            ESS = N_total / (2 τ_corr)
        """
        n = len(samples)
        if n < 10:
            return {"tau_corr": 1.0, "ess": float(n)}

        mean = np.mean(samples)
        var = np.var(samples)
        if var < 1e-14:
            return {"tau_corr": 1.0, "ess": float(n)}

        # Autocorrelation function ρ(k)
        autocorr = []
        for lag in range(min(max_lag, n // 4)):
            cov = np.mean((samples[:n - lag] - mean) * (samples[lag:] - mean))
            autocorr.append(cov / var)

        # Geyer's initial positive sequence truncation
        tau_corr = 1.0
        for k in range(1, len(autocorr)):
            if autocorr[k] <= 0.0:
                break
            tau_corr += 2.0 * autocorr[k]

        tau_corr = max(1.0, tau_corr)
        ess = max(1.0, n / (2.0 * tau_corr))

        return {
            "tau_corr": float(tau_corr),
            "ess": float(ess)
        }
