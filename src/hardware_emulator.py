"""
Project Resonon / PhysLM: Hardware Non-Ideality & Prototype Emulator
====================================================================
Subsystem implementation conforming strictly to RFC-005:
Emulates the physical embodiment of the Quad-Domain Hybrid Testbed:
1. Photonics: Waveguide attenuation (α), dispersion (β2), and optical flight time.
2. Electronics: Memristive conductance non-idealities (write noise, drift, thermal noise).
3. Analog: Johnson-Nyquist thermal noise source.
4. Readout: Square-law photodiode detection, dark current/shot noise, and ADC bit quantization.
5. Metrics: Waveform fidelity ε_ψ and Kullback-Leibler divergence D_KL.
"""

import numpy as np
from typing import Optional, Dict, Any, Tuple


class HardwarePrototypeEmulator:
    def __init__(
        self,
        n_grid: int = 256,
        x_min: float = -10.0,
        x_max: float = 10.0,
        waveguide_length_mm: float = 2.0,
        n_eff: float = 2.45,
        attenuation_db_per_cm: float = 0.5,
        beta2_ps2_per_m: float = -1.2,
        seed: Optional[int] = None
    ):
        self.n_grid = n_grid
        self.x_min = x_min
        self.x_max = x_max
        self.x = np.linspace(x_min, x_max, n_grid, endpoint=False)
        self.dx = float(self.x[1] - self.x[0])
        self.width = float(x_max - x_min)

        self.length_mm = waveguide_length_mm
        self.n_eff = n_eff
        self.c_light = 299792458.0  # m/s
        self.v_g = self.c_light / n_eff

        # Modeled optical transit time: t_flight = L / v_g
        self.t_flight_ps = float((waveguide_length_mm * 1e-3 / self.v_g) * 1e12)

        # Attenuation coefficient: α [1/m] = α [dB/cm] * ln(10) / 10 * 100
        alpha_db_per_m = attenuation_db_per_cm * 100.0
        self.alpha = float(alpha_db_per_m * np.log(10.0) / 10.0)
        self.beta2 = float(beta2_ps2_per_m * 1e-24)  # s^2 / m

        self.rng = np.random.default_rng(seed)

    def propagate_waveguide(
        self,
        psi: np.ndarray,
        phase_noise_std: float = 0.02
    ) -> np.ndarray:
        """
        Emulates physical optical transmission through SOI dielectric waveguide:
        Includes linear propagation loss, engineered GVD phase advance, and laser phase noise.
        """
        length_m = self.length_mm * 1e-3
        # Optical loss: exp(-α L / 2)
        transmission_loss = np.exp(-0.5 * self.alpha * length_m)

        # Phase shift and laser linewidth noise
        laser_phase_jitter = self.rng.normal(0.0, phase_noise_std)
        phase_factor = np.exp(1j * laser_phase_jitter)

        # Frequency domain dispersion
        k = 2.0 * np.pi * np.fft.fftfreq(self.n_grid, d=self.dx)
        dispersion_phase = -0.5 * self.beta2 * (self.v_g ** 2) * (k ** 2) * length_m

        psi_k = np.fft.fft(psi)
        psi_k = psi_k * np.exp(1j * dispersion_phase)
        psi_out = np.fft.ifft(psi_k) * transmission_loss * phase_factor

        return psi_out

    def emulate_memristor_conductance(
        self,
        target_conductance: np.ndarray,
        write_noise_pct: float = 0.03,
        drift_time_sec: float = 10.0,
        drift_exponent: float = 0.04
    ) -> np.ndarray:
        """
        Applies physical memristor non-idealities: programming write noise and time drift.
        G(t) = G0 * (1 + δG_write) * (t/t0)^(-ν)
        """
        g0 = np.array(target_conductance, dtype=float)
        # Programming noise: Gaussian cycle-to-cycle variability
        write_noise = self.rng.normal(0.0, write_noise_pct, size=g0.shape)
        g_written = g0 * (1.0 + write_noise)

        # Temporal relaxation drift
        t0 = 1.0  # reference time: 1 second
        drift_factor = (max(1.0, drift_time_sec) / t0) ** (-drift_exponent)
        g_drifted = g_written * drift_factor

        return np.clip(g_drifted, 1e-6, None)

    def optoelectronic_readout(
        self,
        optical_field: np.ndarray,
        basis_probes: list[np.ndarray],
        adc_bits: int = 8,
        snr_db: float = 35.0
    ) -> np.ndarray:
        """
        Square-law photodiode detection with shot noise and finite ADC bit quantization:
        I_c = |⟨φ_c | ψ⟩|² + noise -> ADC(I_c)
        """
        intensities = []
        for probe in basis_probes:
            # Overlap integral: <φ_c | ψ>
            overlap = np.sum(np.conj(probe) * optical_field) * self.dx
            # Physical square-law power detection
            intensity = float(np.abs(overlap) ** 2)
            intensities.append(intensity)

        raw_intensities = np.array(intensities, dtype=float)

        # Add photodetector thermal/shot noise
        signal_power = float(np.mean(raw_intensities ** 2))
        noise_power = signal_power / (10.0 ** (snr_db / 10.0))
        noise = self.rng.normal(0.0, np.sqrt(max(1e-12, noise_power)), size=len(raw_intensities))
        noisy_intensities = np.clip(raw_intensities + noise, 0.0, None)

        # Uniform ADC bit quantization
        max_val = float(np.max(noisy_intensities))
        if max_val > 1e-9:
            num_levels = (1 << adc_bits) - 1
            quantized = np.round((noisy_intensities / max_val) * num_levels) / num_levels * max_val
        else:
            quantized = noisy_intensities

        # Normalize to probability-like distribution
        total = np.sum(quantized)
        if total > 1e-12:
            return quantized / total
        return np.ones_like(quantized) / len(quantized)

    def compute_waveform_error(self, psi_hw: np.ndarray, psi_sim: np.ndarray) -> float:
        """Computes Hilbert norm discrepancy: ε_ψ = ||ψ_hw - ψ_sim||."""
        diff = psi_hw - psi_sim
        return float(np.sqrt(np.sum(np.abs(diff) ** 2) * self.dx))

    def compute_kl_divergence(self, p_hw: np.ndarray, p_sim: np.ndarray) -> float:
        """Computes Kullback-Leibler divergence D_KL(P_hw || P_sim)."""
        p = np.array(p_hw, dtype=float) + 1e-15
        q = np.array(p_sim, dtype=float) + 1e-15
        p = p / np.sum(p)
        q = q / np.sum(q)
        return float(np.sum(p * np.log(p / q)))
