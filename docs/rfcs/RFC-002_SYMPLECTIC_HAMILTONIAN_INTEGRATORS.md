# RFC-002: Symplectic Hamiltonian Integrators
## Subsystem Specification: Unitary Wave Propagation & Dissipative Relaxation

* **Status:** `ACTIVE` (Implemented & Verified in [`src/baseline_phase0.py`](../../src/baseline_phase0.py))
* **Author:** Project Resonon / PhysLM
* **Base Document:** [Mathematical Formulation 02](../backbone/02_MATHEMATICAL_AND_PHYSICAL_FORMULATION.md)

---

## 1. Problem Statement

Standard non-symplectic numerical integrators (e.g. classical Runge-Kutta RK4) fail to conserve the symplectic 2-form of Hamiltonian dynamical systems. On continuous complex wavefields with high spatial frequencies:
1. RK4 introduces artificial numerical damping $|R(i y)| < 1$ along the imaginary axis, slowly destroying the total probability norm $N = \int |\psi|^2 dx$.
2. In non-linear regimes ($g \neq 0$), finite-difference Laplacians create numerical dispersion and phase error.

---

## 2. Specification: Strang Split-Operator FFT Integration

RFC-002 specifies the **Symplectic Strang Split-Operator Spectral Method** for real-time unitary wave evolution:

$$i \hbar \frac{\partial \psi(\mathbf{x}, t)}{\partial t} = \left[ \hat{T} + \hat{V}(\mathbf{x}, t) + g |\psi(\mathbf{x}, t)|^2 \right] \psi(\mathbf{x}, t)$$

### 2.1 Operator Decomposition
The continuous time evolution operator $e^{-i \hat{H} \Delta t / \hbar}$ is split symmetrically:

$$\hat{U}(\Delta t) = \exp\left( -i \frac{\hat{V}_{\text{eff}} \Delta t}{2 \hbar} \right) \exp\left( -i \frac{\hat{T} \Delta t}{\hbar} \right) \exp\left( -i \frac{\hat{V}_{\text{eff}} \Delta t}{2 \hbar} \right) + \mathcal{O}(\Delta t^3)$$

where $\hat{V}_{\text{eff}}(\mathbf{x}) = V(\mathbf{x}) + g |\psi(\mathbf{x})|^2$.

### 2.2 Kinetic Step in Fourier Spectral Space
The kinetic operator $\hat{T} = -\frac{\hbar^2}{2m} \nabla^2$ is evaluated diagonally in reciprocal space via the Fast Fourier Transform:

$$\exp\left( -i \frac{\hat{T} \Delta t}{\hbar} \right) \psi = \mathcal{F}^{-1} \left\{ \exp\left( -i \frac{\hbar \|\mathbf{k}\|^2 \Delta t}{2 m} \right) \mathcal{F}\{\psi\} \right\}$$

Because $|\exp(-i \theta)| \equiv 1$ identically for every frequency mode $\mathbf{k}$, **the probability norm is preserved unconditionally to floating-point machine precision ($< 10^{-14}$)** without numerical leakage.

---

## 3. Dissipative Relaxation for Attractor Settlement

For thermodynamic energy minimization (settling into learned semantic potential wells $V(\mathbf{x})$):
$$\hbar \frac{\partial \psi}{\partial \tau} = -\left[ \hat{H}[\psi] - E_{\text{ref}} \right] \psi + \xi_{\text{thermal}}(\tau)$$

This gradient flow drives the wave packet into the closest local minimum of the continuous Hopfield landscape.
