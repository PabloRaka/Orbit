"""
Project Resonon / PhysLM: Thermal / Boltzmann Engine
====================================================
Subsystem 4 implementation conforming strictly to RFC-004:
    dψ = -μ ∇E(ψ) dt + √(2μ k_B T) dW_t

Governs:
- Overdamped complex Langevin stochastic differential equations
- Complex Wiener increment generation conforming to Fluctuation-Dissipation Theorem
- Stationary Gibbs-Boltzmann equilibrium distribution P(ψ) ∝ exp(-E / k_B T)
- Thermodynamic Boltzmann sampling without digital softmax
"""

import numpy as np
from typing import Optional, Dict, Any, Tuple


class ThermalBoltzmannEngine:
    def __init__(
        self,
        n_grid: int = 256,
        x_min: float = -10.0,
        x_max: float = 10.0,
        mobility: float = 1.0,
        k_b: float = 1.0,
        seed: Optional[int] = None
    ):
        self.n_grid = n_grid
        self.x_min = x_min
        self.x_max = x_max
        self.x = np.linspace(x_min, x_max, n_grid, endpoint=False)
        self.dx = float(self.x[1] - self.x[0])
        self.width = float(x_max - x_min)

        self.mobility = float(mobility)
        self.k_b = float(k_b)
        self.rng = np.random.default_rng(seed)

    def generate_wiener_increment(self, dt: float, temperature: float) -> np.ndarray:
        """
        Generates complex Wiener thermal fluctuation increment:
            η(x) = √(μ k_B T dt / dx) * (ξ_R + i ξ_I)
        Satisfies Fluctuation-Dissipation: Var(η) = 2μ k_B T dt / dx.
        """
        if temperature <= 0.0:
            return np.zeros(self.n_grid, dtype=complex)

        std_per_channel = np.sqrt(self.mobility * self.k_b * temperature * dt / self.dx)
        xi_r = self.rng.normal(0.0, std_per_channel, self.n_grid)
        xi_i = self.rng.normal(0.0, std_per_channel, self.n_grid)
        return xi_r + 1j * xi_i

    def step_langevin(
        self,
        psi: np.ndarray,
        grad_energy: np.ndarray,
        dt: float,
        temperature: float = 0.1,
        renormalize: bool = True
    ) -> np.ndarray:
        """
        Single Euler-Maruyama step for overdamped complex Langevin SDE:
            ψ_{n+1} = ψ_n - μ ∇E(ψ_n) dt + η_n
        """
        deterministic_drift = -self.mobility * grad_energy * dt
        thermal_noise = self.generate_wiener_increment(dt, temperature)

        psi_next = psi + deterministic_drift + thermal_noise

        if renormalize:
            norm = np.sqrt(np.sum(np.abs(psi_next) ** 2) * self.dx)
            if norm > 1e-12:
                psi_next = psi_next / norm

        return psi_next

    def boltzmann_ratio(self, e_a: float, e_b: float, temperature: float) -> float:
        """Computes theoretical equilibrium ratio: P(A) / P(B) = exp(-(E_A - E_B) / (k_B T))."""
        if temperature <= 0.0:
            return float("inf") if e_a < e_b else 0.0
        return float(np.exp(-(e_a - e_b) / (self.k_b * temperature)))

    def compute_entropy(self, probabilities: np.ndarray) -> float:
        """Computes Shannon entropy H = -Σ p ln(p)."""
        p = np.array(probabilities, dtype=float)
        p = p[p > 1e-15]
        return float(-np.sum(p * np.log(p)))

    def check_stability(self, psi: np.ndarray) -> Dict[str, Any]:
        """Verifies state is finite without NaN/Inf anomalies."""
        if np.any(np.isnan(psi)) or np.any(np.isinf(psi)):
            raise FloatingPointError("Numerical instability: NaN or Inf in thermal state.")
        norm = float(np.sum(np.abs(psi) ** 2) * self.dx)
        return {"norm": norm, "is_stable": True}
