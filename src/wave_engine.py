"""
Project Resonon / PhysLM: Wave Dynamics Engine
==============================================
Subsystem 2 specification implementation conforming strictly to RFC-003:
    i ∂ψ/∂t = -β ∂²ψ/∂x² + g|ψ|²ψ - iγψ + V(x)ψ + F(x,t)

Supports:
- Mode A (Conservative / Free Flight): γ = 0, F = 0
- Mode B (Dissipative Attractor): γ > 0, V = V_attractor
- Mode C (Forced / Coupled): F(x,t) ≠ 0
- Primary Solver: Strang Split-Step Fourier Method (O(N log N))
- Reference Solver: 4th-order Runge-Kutta (RK4)
"""

import numpy as np
from typing import Optional, Dict, Any, Tuple


class WaveDynamicsEngine:
    def __init__(
        self,
        n_grid: int = 256,
        x_min: float = -10.0,
        x_max: float = 10.0,
        beta: float = 0.5,
        g: float = 0.0,
        gamma: float = 0.0,
        boundary: str = "periodic"
    ):
        self.n_grid = n_grid
        self.x_min = x_min
        self.x_max = x_max
        self.x = np.linspace(x_min, x_max, n_grid, endpoint=False)
        self.dx = float(self.x[1] - self.x[0])
        self.width = float(x_max - x_min)

        self.beta = float(beta)
        self.g = float(g)
        self.gamma = float(gamma)
        self.boundary = boundary

        # Fourier wavenumber lattice
        self.k = 2.0 * np.pi * np.fft.fftfreq(n_grid, d=self.dx)

    def laplacian(self, psi: np.ndarray) -> np.ndarray:
        """Central difference spatial second derivative with declared boundary."""
        if self.boundary == "periodic":
            return (np.roll(psi, -1) - 2.0 * psi + np.roll(psi, 1)) / (self.dx ** 2)
        elif self.boundary == "dirichlet":
            lap = np.zeros_like(psi)
            lap[1:-1] = (psi[2:] - 2.0 * psi[1:-1] + psi[:-2]) / (self.dx ** 2)
            lap[0] = (psi[1] - 2.0 * psi[0]) / (self.dx ** 2)
            lap[-1] = (psi[-2] - 2.0 * psi[-1]) / (self.dx ** 2)
            return lap
        else:
            raise ValueError(f"Unknown boundary condition: {self.boundary}")

    def compute_norm(self, psi: np.ndarray) -> float:
        """Computes probability norm: int |psi(x)|^2 dx."""
        return float(np.sum(np.abs(psi) ** 2) * self.dx)

    def compute_energy(self, psi: np.ndarray, potential: Optional[np.ndarray] = None) -> float:
        """Computes Hamiltonian energy functional."""
        dpsi_dx = (np.roll(psi, -1) - np.roll(psi, 1)) / (2.0 * self.dx)
        kinetic = self.beta * np.abs(dpsi_dx) ** 2
        pot = (potential * np.abs(psi) ** 2) if potential is not None else 0.0
        interaction = 0.5 * self.g * (np.abs(psi) ** 4)
        return float(np.real(np.sum(kinetic + pot + interaction) * self.dx))

    def step_split_step(
        self,
        psi: np.ndarray,
        potential: Optional[np.ndarray] = None,
        forcing: Optional[np.ndarray] = None,
        dt: float = 0.001
    ) -> np.ndarray:
        """
        Strang Split-Step Fourier Integrator:
        Linear kinetic + dissipation half-step -> Non-linear potential full-step -> Linear half-step.
        """
        # Half-step linear kinetic & dissipation in Fourier space
        psi_k = np.fft.fft(psi)
        l_half = (-1j * self.beta * (self.k ** 2) - self.gamma) * (0.5 * dt)
        psi_k = psi_k * np.exp(l_half)
        psi_mid = np.fft.ifft(psi_k)

        # Full-step potential & non-linear Kerr rotation in real space
        v_eff = self.g * (np.abs(psi_mid) ** 2)
        if potential is not None:
            v_eff = v_eff + potential

        n_full = -1j * v_eff * dt
        psi_mid = psi_mid * np.exp(n_full)
        if forcing is not None:
            psi_mid = psi_mid + forcing * dt

        # Final half-step linear kinetic & dissipation in Fourier space
        psi_k_final = np.fft.fft(psi_mid)
        psi_k_final = psi_k_final * np.exp(l_half)
        psi_next = np.fft.ifft(psi_k_final)

        return psi_next

    def derivative(
        self,
        psi: np.ndarray,
        potential: Optional[np.ndarray] = None,
        forcing: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Computes d(psi)/dt according to governing wave PDE."""
        kinetic = -self.beta * self.laplacian(psi)
        nonlin = self.g * (np.abs(psi) ** 2) * psi
        pot_term = (potential * psi) if potential is not None else 0.0
        force_term = forcing if forcing is not None else 0.0

        h_psi = kinetic + nonlin + pot_term + force_term
        dpsi_dt = -1j * h_psi - self.gamma * psi
        return dpsi_dt

    def step_rk4(
        self,
        psi: np.ndarray,
        potential: Optional[np.ndarray] = None,
        forcing: Optional[np.ndarray] = None,
        dt: float = 0.001
    ) -> np.ndarray:
        """Reference 4th-order Runge-Kutta integrator for cross-validation."""
        k1 = self.derivative(psi, potential, forcing)
        k2 = self.derivative(psi + 0.5 * dt * k1, potential, forcing)
        k3 = self.derivative(psi + 0.5 * dt * k2, potential, forcing)
        k4 = self.derivative(psi + dt * k3, potential, forcing)
        return psi + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    def check_stability(self, psi: np.ndarray, initial_norm: float = 1.0) -> Dict[str, Any]:
        """Validates numerical integrity and raises on instability anomalies."""
        if np.any(np.isnan(psi)) or np.any(np.isinf(psi)):
            raise FloatingPointError("Numerical instability: NaN or Inf detected in wave state.")

        norm = self.compute_norm(psi)
        if self.gamma == 0.0 and abs(norm - initial_norm) > 0.05:
            raise FloatingPointError(f"Norm explosion detected: norm={norm:.6f}, expected={initial_norm:.6f}")

        max_amp = float(np.max(np.abs(psi)))
        return {
            "norm": norm,
            "max_amplitude": max_amp,
            "is_stable": True
        }
