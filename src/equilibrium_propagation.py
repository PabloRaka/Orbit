"""
Project Resonon / PhysLM: Equilibrium Propagation Engine (No-Backprop Learning)
================================================================================
Implements local, energy-based parameter adaptation for physical memristive
crossbars without reverse computational graphs or backpropagation.

Formulation (Scellier & Bengio, 2017 adapted to Complex Hilbert States):
    1. Phase-Preserving Complex Saturation:
       f(z) = z / (1.0 + |z|)
       Maintains relational phase arg(z) while bounding amplitude to [0, 1).

    2. Energy Landscape:
       E(h, y) = 0.5*||h||^2 + 0.5*||y||^2 - Re( h^H * W_in * x + 0.5 * h^H * W_rec * h + y^H * W_out * h )

    3. Free Phase (tau in [0, T_free]):
       Contractive relaxation to free equilibrium (h^0, y^0) under input clamp x.
       dh/dtau = f(W_in * x + W_rec * h + W_out^H * y) - h
       dy/dtau = f(W_out * h) - y

    4. Nudged Phase (tau in [0, T_nudge]):
       Perturbed by target y* with small nudging parameter beta:
       dy/dtau = f(W_out * h) - y + beta * (y* - y)
       Settles to nudged equilibrium (h^beta, y^beta).

    5. Local Parameter Update (Contrastive Hebbian Rule on Crossbar):
       Delta W = (eta / beta) * ( s^beta * (s^beta)^H - s^0 * (s^0)^H )
       Occurs purely through local energy relaxation at each physical wire junction.
"""

import numpy as np
from typing import Tuple, Dict, Any


def phase_preserving_saturation(z: np.ndarray) -> np.ndarray:
    """Soft amplitude saturation f(z) = z / (1 + |z|) preserving complex phase."""
    abs_z = np.abs(z)
    scale = 1.0 / (1.0 + abs_z)
    return z * scale


class MemristiveCrossbarNetwork:
    def __init__(
        self,
        dim_in: int,
        dim_hid: int,
        dim_out: int,
        eta: float = 0.03,
        beta: float = 0.3,
        dt: float = 0.15
    ):
        self.dim_in = dim_in
        self.dim_hid = dim_hid
        self.dim_out = dim_out
        self.eta = eta
        self.beta = beta
        self.dt = dt

        # Contractive weight initialization (Lipschitz bound < 1.0 ensures fixed point)
        self.W_in = 0.5 * (np.random.normal(0, 1, (dim_hid, dim_in)) + 
                           1j * np.random.normal(0, 1, (dim_hid, dim_in))) / np.sqrt(dim_in)

        w_rnd = 0.25 * (np.random.normal(0, 1, (dim_hid, dim_hid)) + 
                         1j * np.random.normal(0, 1, (dim_hid, dim_hid))) / np.sqrt(dim_hid)
        self.W_rec = 0.5 * (w_rnd + np.conj(w_rnd).T)

        self.W_out = 0.4 * (np.random.normal(0, 1, (dim_out, dim_hid)) + 
                            1j * np.random.normal(0, 1, (dim_out, dim_hid))) / np.sqrt(dim_hid)

    def relax(
        self,
        x: np.ndarray,
        target: np.ndarray = None,
        h_init: np.ndarray = None,
        y_init: np.ndarray = None,
        steps: int = 35
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulates continuous physical relaxation toward energy equilibrium.
        If target is None -> Free Phase.
        If target is provided -> Nudged Phase with coupling beta.
        """
        h = np.zeros(self.dim_hid, dtype=complex) if h_init is None else h_init.copy()
        y = np.zeros(self.dim_out, dtype=complex) if y_init is None else y_init.copy()

        for _ in range(steps):
            # Currents arriving at hidden nodes
            current_h = self.W_in @ x + self.W_rec @ h + np.conj(self.W_out).T @ y
            dh = phase_preserving_saturation(current_h) - h

            # Currents arriving at output nodes
            current_y = self.W_out @ h
            dy = phase_preserving_saturation(current_y) - y

            if target is not None:
                # Nudging force pulling output towards target
                dy += self.beta * (target - y)

            h += self.dt * dh
            y += self.dt * dy

        return h, y

    def train_step(self, x: np.ndarray, target: np.ndarray, free_steps: int = 30, nudge_steps: int = 15) -> float:
        """
        Executes one complete Equilibrium Propagation step:
        Free Phase -> Nudged Phase -> Local Parameter Adjustment.
        Returns MSE loss at free equilibrium.
        """
        # 1. Free Phase
        h_0, y_0 = self.relax(x, target=None, steps=free_steps)
        loss = float(np.mean(np.abs(y_0 - target) ** 2))

        # 2. Nudged Phase (initialized from free equilibrium state)
        h_beta, y_beta = self.relax(x, target=target, h_init=h_0, y_init=y_0, steps=nudge_steps)

        # 3. Local Contrastive Parameter Updates (Delta W = (eta/beta) * (nudged - free))
        coeff = self.eta / self.beta

        dW_out = coeff * (np.outer(y_beta, np.conj(h_beta)) - np.outer(y_0, np.conj(h_0)))
        dW_in = coeff * (np.outer(h_beta, np.conj(x)) - np.outer(h_0, np.conj(x)))
        dW_rec = coeff * (np.outer(h_beta, np.conj(h_beta)) - np.outer(h_0, np.conj(h_0)))

        # Update physical conductances
        self.W_out += dW_out
        self.W_in += dW_in
        # Maintain Hermitian symmetry of recurrent reservoir
        dW_rec_sym = 0.5 * (dW_rec + np.conj(dW_rec).T)
        self.W_rec += dW_rec_sym

        return loss

    def predict(self, x: np.ndarray, steps: int = 30) -> np.ndarray:
        """Runs inference via free physical relaxation."""
        _, y = self.relax(x, target=None, steps=steps)
        return y
