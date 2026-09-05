# RFC-003: Wave Dynamics Engine Specification
## Phase II Architectural Contract: Governing Wave Equation, Solvers, Regimes, and Stability Conditions

* **Status:** `RATIFIED ARCHITECTURAL CONTRACT` (Phase II Architecture Consolidation)
* **Author:** Project Resonon / PhysLM Core Architecture Group
* **Scope:** Physical Evolution Operator $\mathcal{D}_{\Delta t}: \mathcal{H} \to \mathcal{H}$, Governing PDE, Linear Dispersion, Kerr Non-linearity, Dissipation, Cavity Resonators, Split-Step & RK4 Solvers, and Normative Conformance Tests
* **Parent Architecture:** [RFC-001: PhysLM Core Architecture](RFC-001_PHYSLM_CORE_ARCHITECTURE.md)
* **State Contract:** [RFC-002: Continuous Hilbert State Specification](RFC-002_CONTINUOUS_HILBERT_STATE.md)
* **Empirical Ground Truth:** [`docs/benchmarks/10_EXPERIMENTAL_BASELINE_FREEZE.md`](../benchmarks/10_EXPERIMENTAL_BASELINE_FREEZE.md)

---

## 1. Scope & Design Invariants

RFC-003 specifies the **Wave Dynamics Engine** (Subsystem 2 of PhysLM).

```text
+-------------------------------------------------------------------------------+
|                       WAVE DYNAMICS ENGINE BOUNDARY                           |
+-------------------------------------------------------------------------------+
|                                                                               |
|  Input Contract:   |ψ(x, t0)⟩ ∈ H = L²(Ω, C)                                  |
|  Operator:         D_Δt : H → H                                               |
|  Output Contract:  |ψ(x, t0 + Δt)⟩ ∈ H = L²(Ω, C)                             |
|                                                                               |
|  Strict Exclusions: NO token IDs, NO KV-cache, NO hidden history buffers,     |
|                     NO global backpropagation, NO weight learning.            |
|                                                                               |
+-------------------------------------------------------------------------------+
```

### Core Architectural Principle:
> **The Wave Engine is not an LLM.**  
> It is a **continuous dynamical substrate**. Language processing capabilities arise from the structured coupling between the Continuous Hilbert State (RFC-002), the Wave Dynamics Engine (RFC-003), the Associative Attractor Crossbar (RFC-001), and the Measurement Interface (RFC-002).

---

## 2. Governing Wave Equation

The continuous physical evolution of the wavefield $\psi(x,t) \in \mathcal{H}$ is governed by the Non-linear Ginzburg-Landau / Gross-Pitaevskii differential equation:

$$\boxed{i \frac{\partial\psi(x,t)}{\partial t} = -\beta \frac{\partial^2\psi(x,t)}{\partial x^2} + g |\psi(x,t)|^2 \psi(x,t) - i \gamma \psi(x,t) + V(x) \psi(x,t) + F(x,t)}$$

where:
- $\beta \in \mathbb{R}^+$: Kinetic dispersion coefficient ($\beta = \hbar^2 / (2m)$ in quantum mechanics, or $\beta = 1 / (2 k_0)$ in paraxial beam optics).
- $g \in \mathbb{R}$: Kerr / cubic non-linear interaction coefficient.
- $\gamma \ge 0$: Linear dissipative loss coefficient.
- $V(x) \in \mathbb{R}$: Spatial confining potential / cavity refractive index profile.
- $F(x,t) \in \mathbb{C}$: External driving source (prompt injection, Dyck cavity modes, or crossbar coupling).

### Three Operating Regimes

```text
+-------------------------------------------------------------------------------+
|                        THREE OPERATING DYNAMICAL REGIMES                      |
+-------------------------------------------------------------------------------+
|                                                                               |
|  MODE A — Conservative / Free Flight:                                         |
|    γ = 0, F(x,t) = 0                                                          |
|    Pure Hamiltonian evolution. Unitary norm and energy are strictly conserved.|
|    Used for causal rollout, interference, and measuring long-horizon drift.   |
|                                                                               |
|  MODE B — Dissipative Attractor:                                              |
|    γ > 0, V(x) = V_attractor(x)                                               |
|    Open thermodynamic relaxation. Drives state toward nearest energy minimum. |
|    Used for semantic associative infilling and equilibrium stabilization.     |
|                                                                               |
|  MODE C — Forced / Coupled:                                                   |
|    F(x,t) ≠ 0                                                                 |
|    Non-homogeneous driving. Injects input prompts, Dyck grammar modes, or     |
|    Equilibrium Propagation nudge fields.                                      |
|                                                                               |
+-------------------------------------------------------------------------------+
```

---

## 3. Linear Dispersion ($-\beta \partial_{xx}$)

The differential operator $-\beta \frac{\partial^2}{\partial x^2}$ generates spatial wave spreading according to the parabolic dispersion relation:

$$\omega(k) = \beta k^2$$

### Analytical Plane Wave Solution:
For a spatial Fourier mode $\psi_k(x, 0) = A e^{i k x}$ under pure dispersion ($\gamma = 0, g = 0, V = 0$):

$$\psi_k(x, t) = A \exp\left( i (k x - \beta k^2 t) \right)$$

### Role in Language Processing:
Dispersion spreads localized character wave packets across adjacent spatial coordinate slots ($x_j \to x_{j \pm 1}$), mediating natural context mixing and interference across tokens **without requiring digital attention matrices or quadratic all-to-all dot products**.

---

## 4. Kerr / Non-linear Interaction ($g |\psi|^2 \psi$)

The cubic non-linear term introduces self-phase modulation proportional to local wave intensity $I(x,t) = |\psi(x,t)|^2$:

$$\Delta \phi_{\text{nonlin}}(x, t) = -g |\psi(x,t)|^2 \Delta t$$

- **Focusing Non-linearity ($g < 0$)**: Compresses wave packets, counteracting dispersion to form localized, self-reinforcing solitons.
- **Defocusing Non-linearity ($g > 0$)**: Broadens wave packets and mediates repulsive wave-wave interactions.
- In PhysLM, $g$ is calibrated to maintain packet integrity while allowing semantic phase-locking during crossbar relaxation.

---

## 5. Dissipation ($-i \gamma \psi$) & Conservation Dichotomy

PhysLM establishes a strict mathematical distinction between conservative and dissipative regimes:

### 5.1 Conservative Regime ($\gamma = 0$)
Under zero dissipation, the system is strictly Hamiltonian:
$$\frac{d}{dt} \|\psi(t)\|^2 = 0, \quad \frac{d}{dt} E[\psi(t)] = 0$$
Symplectic conservation holds to machine precision ($|\Delta N| < 10^{-14}$, $|\Delta E/E_0| < 10^{-5}$).

### 5.2 Dissipative Regime ($\gamma > 0$)
Under non-zero dissipation, the system is strictly non-conservative. Norm and energy decay monotonically:

$$\boxed{\frac{d}{dt} \|\psi(t)\|^2 = -2\gamma \|\psi(t)\|^2 \implies \|\psi(t)\|^2 = \|\psi(0)\|^2 e^{-2\gamma t}}$$

> [!WARNING]
> **No False Symplectic Claims**: When $\gamma > 0$, solvers and benchmarks **must not claim symplectic energy conservation**. The normative invariant in Mode B is strictly monotonic, controlled energy dissipation toward equilibrium.

---

## 6. Cavity & Boundary Conditions

All wave evolution occurs on the bounded domain $\Omega = [-W/2, W/2]$. The boundary conditions must be declared explicitly:

### 6.1 Dirichlet Boundary (Reflective Resonator Cavity)
$$\psi(-W/2, t) = \psi(W/2, t) = 0 \quad \forall t$$
- Models hard optical mirrors or electronic cavity boundaries.
- Discrete eigenmodes form standing waves:
  $$\chi_n(x) = \sqrt{\frac{2}{W}} \sin\left( \frac{n \pi (x + W/2)}{W} \right), \quad n \in \{1, 2, \dots\}$$
- Utilized in the **Stackless Dyck Grammar Cavity** (Subsystem 2 / RFC-001) where harmonic modes $n \le 16$ track nesting levels.

### 6.2 Periodic Boundary (Ring Resonator / Spectral Lattice)
$$\psi(-W/2, t) = \psi(W/2, t), \quad \partial_x \psi(-W/2, t) = \partial_x \psi(W/2, t)$$
- Models continuous closed optical loop resonators.
- Default boundary condition for Split-Step Fourier spectral transforms.

---

## 7. Continuous Attractor Coupling

The wavefield couples to external energy landscapes via potential $V(x)$ and forcing $F(x,t)$:

### 7.1 Confining Potential
$$V_{\text{trap}}(x) = \frac{1}{2} m \omega_0^2 x^2$$
Prevents spatial wave packet leakage beyond the operational containment region $[-W/2, W/2]$.

### 7.2 Associative Memory Coupling
$$V_{\text{Hopfield}}(x) = -\sum_{\mu=1}^{P} \xi_\mu(x) \text{Re}\langle \xi_\mu | \psi \rangle$$
Shapes the potential surface into multi-well attractor basins corresponding to stored linguistic concepts.

### 7.3 Hierarchical Dyck Forcing
$$F_{\text{Dyck}}(x, t) = \sum_{m=1}^{D_{\max}} a_m(t) \chi_m(x)$$
Injects orthogonal cavity modes during parenthesis open/close transitions.

---

## 8. Numerical Integrators (Primary & Reference)

To guarantee numerical integrity, PhysLM specifies both a primary production solver and an independent reference solver:

### 8.1 Primary Solver: Strang Split-Step Fourier Method
Decomposes the evolution operator into linear kinetic/dissipative step $\mathcal{L}$ and non-linear potential step $\mathcal{N}$:

$$\psi_{t+\Delta t} = \exp\left( \frac{\mathcal{L} \Delta t}{2} \right) \exp\left( \mathcal{N} \Delta t \right) \exp\left( \frac{\mathcal{L} \Delta t}{2} \right) \psi_t + \mathcal{O}(\Delta t^3)$$

1. **Linear Kinetic & Dissipative Half-Step (Fourier Space)**:
   $$\hat{\psi}(k) \longleftarrow \hat{\psi}(k) \cdot \exp\left( \left( -i \beta k^2 - \gamma \right) \frac{\Delta t}{2} \right)$$
2. **Non-linear & Potential Full-Step (Real Space)**:
   $$V_{\text{eff}}(x) = V(x) + g |\psi(x)|^2$$
   $$\psi(x) \longleftarrow \psi(x) \cdot \exp\left( -i V_{\text{eff}}(x) \Delta t \right) + F(x, t) \Delta t$$
3. **Linear Kinetic & Dissipative Final Half-Step (Fourier Space)**:
   $$\hat{\psi}(k) \longleftarrow \hat{\psi}(k) \cdot \exp\left( \left( -i \beta k^2 - \gamma \right) \frac{\Delta t}{2} \right)$$

- **Complexity**: Exactly $\mathcal{O}(N_{\text{grid}} \log N_{\text{grid}})$ via FFT.
- **Stability**: Unconditionally norm-stable; zero artificial numerical dispersion.

### 8.2 Reference Solver: 4th-Order Runge-Kutta (RK4)
Used for independent numerical cross-validation on equivalent timesteps:

$$\frac{\partial\psi}{\partial t} = \mathcal{F}(\psi) = -i \left( -\beta \partial_{xx} \psi + g |\psi|^2 \psi + V(x) \psi + F(x,t) \right) - \gamma \psi$$

$$\begin{aligned}
k_1 &= \mathcal{F}(\psi_t) \\
k_2 &= \mathcal{F}\left(\psi_t + \frac{\Delta t}{2} k_1\right) \\
k_3 &= \mathcal{F}\left(\psi_t + \frac{\Delta t}{2} k_2\right) \\
k_4 &= \mathcal{F}(\psi_t + \Delta t \, k_3) \\
\psi_{t+\Delta t} &= \psi_t + \frac{\Delta t}{6} (k_1 + 2k_2 + 2k_3 + k_4)
\end{aligned}$$

---

## 9. Stability & Conservation Conditions

### 9.1 Mandatory Step Telemetry
At every simulation step $t_k$, the engine must compute and log:
1. **Probability Norm**: $N(t) = \int_\Omega |\psi(x,t)|^2 dx$
2. **Hamiltonian Energy**: $E(t) = \int_\Omega \left[ \beta |\partial_x \psi|^2 + V(x)|\psi|^2 + \frac{1}{2}g |\psi|^4 \right] dx$
3. **Peak Amplitude**: $A_{\max}(t) = \max_{x \in \Omega} |\psi(x,t)|$
4. **Step Increment**: $\Delta \psi_t = \|\psi_{t+\Delta t} - \psi_t\|$

### 9.2 Anomaly & Instability Detection
The solver must abort and raise an exception if any of the following conditions occur:
- $\text{isnan}(\psi)$ or $\text{isinf}(\psi)$ at any lattice site.
- Norm explosion: $|N(t) - 1.0| > 0.05$ (in Mode A).
- Amplitude divergence: $A_{\max}(t) > 10.0 \times A_{\max}(0)$.
- Phase instability: high-frequency spatial oscillations approaching Nyquist limit $k_{\max} = \pi / \Delta x$.

---

## 10. Measurement-Free Free-Flight Mode

In Mode A ($\gamma = 0, F = 0$), the Wave Dynamics Engine propagates wave packets continuously without Born-rule projective measurement.

### Key Finding from Phase I:
$$\boxed{\text{Long-Horizon Analog Drift: } L(256) = 2.0217, \quad R_\phi(256) = 0.2769}$$
- Uncollapsed waves drift toward the orthogonal random saturation limit ($L \to 2.0$) over long horizons $H$.
- The Wave Engine's responsibility is to execute free-flight propagation faithfully so downstream modules can measure phase decoherence and trigger projective restoration (Mode B / RFC-001) when $R_\phi$ falls below threshold $\theta_{\text{proj}} = 0.50$.

---

## 11. Interface to EqProp & Thermal Engine

```text
+-------------------------------------------------------------------------------+
|                      SUBSYSTEM HANDOVER CONTRACTS                             |
+-------------------------------------------------------------------------------+
|                                                                               |
|  Wave Dynamics → Equilibrium Propagation (Subsystem 3 / RFC-001):             |
|    The Wave Engine integrates the state until convergence:                   |
|      ||ψ_{t+Δt} - ψ_t|| < ε_relax                                            |
|    Returns equilibrium state |ψ^0⟩ (free phase) and |ψ^β⟩ (clamped phase).    |
|    The crossbar weights G_ij are updated EXTERNALLY by the EqProp module.    |
|                                                                               |
|  Thermal Engine → Wave Dynamics (Subsystem 4 / RFC-004):                      |
|    Thermal Engine provides stochastic noise field:                           |
|      F(x, t) = √(2 k_B T) ξ(x, t)                                             |
|    Wave Engine integrates the resulting Langevin SDE:                         |
|      dψ = -i H[ψ] dt - γ ψ dt + F(x, t) dt                                    |
|                                                                               |
+-------------------------------------------------------------------------------+
```

---

## 12. Normative Conformance Tests

Compliant implementations of the Wave Dynamics Engine must satisfy all eight normative test cases:

```text
====================================================================================================
RFC-003 NORMATIVE CONFORMANCE SUITE
====================================================================================================
Test ID    Name                              Pass Condition
----------------------------------------------------------------------------------------------------
WAVE-001   Analytical Dispersion Match       Phase advance error |Δφ_num(k) - β k² Δt| < 10^-4
WAVE-002   Conservative Norm Conservation    | ||ψ(t)||^2 - ||ψ(0)||^2 | < 10^-10 at γ = 0
WAVE-003   Controlled Dissipation Decay      ||ψ(t)||^2 = ||ψ(0)||^2 e^(-2γ t) ± 10^-4 at γ > 0
WAVE-004   Numerical Horizon Stability       Zero NaN / Inf over 500 consecutive integration steps
WAVE-005   Lattice Refinement Convergence    ||ψ_fine - ψ_ref|| < ||ψ_coarse - ψ_ref||
WAVE-006   Split-Step vs RK4 Agreement       dH(ψ_split, ψ_rk4) < 10^-3 over equivalent Δt
WAVE-007   Cavity Eigenmode Stability        Stationary harmonic mode profile preserved over time
WAVE-008   Nonlinear Amplitude Bound         Peak amplitude remains bounded: max |ψ(x,t)| < M_bound
====================================================================================================
```

These normative test cases are automated in [`tests/test_rfc003_conformance.py`](../../tests/test_rfc003_conformance.py).
