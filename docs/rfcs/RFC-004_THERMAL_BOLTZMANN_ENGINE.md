# RFC-004: Thermal / Boltzmann Engine Specification
## Phase II Architectural Contract: Langevin Stochastic Dynamics, Fokker-Planck Equilibrium, and Fluctuation-Dissipation Relations

* **Status:** `RATIFIED ARCHITECTURAL CONTRACT` (Phase II Architecture Consolidation)
* **Author:** Project Resonon / PhysLM Core Architecture Group
* **Scope:** Physical Stochastic SDE Operator $\mathcal{D}_{\Delta t}^{T}$, Free-Energy Landscapes, Overdamped Complex Langevin Equations, Fokker-Planck Equation, Fluctuation-Dissipation Theorem, Thermal Boltzmann Sampling, and Normative Conformance Tests
* **Parent Architecture:** [RFC-001: PhysLM Core Architecture](RFC-001_PHYSLM_CORE_ARCHITECTURE.md)
* **State Contract:** [RFC-002: Continuous Hilbert State Specification](RFC-002_CONTINUOUS_HILBERT_STATE.md)
* **Deterministic Dynamics:** [RFC-003: Wave Dynamics Engine Specification](RFC-003_WAVE_DYNAMICS_ENGINE.md)
* **Empirical Ground Truth:** [`docs/benchmarks/10_EXPERIMENTAL_BASELINE_FREEZE.md`](../benchmarks/10_EXPERIMENTAL_BASELINE_FREEZE.md)

---

## 1. Scope & Thermodynamic Interpretation

RFC-004 specifies the **Thermal / Boltzmann Engine** (Subsystem 4 of PhysLM).

```text
+-------------------------------------------------------------------------------+
|                       THERMAL / BOLTZMANN ENGINE BOUNDARY                     |
+-------------------------------------------------------------------------------+
|                                                                               |
|  Deterministic Evolution:   D_Δt : H → H  (from RFC-003)                      |
|  Stochastic SDE Operator:   D_Δt^T : (ψ, E, T_eff) → ψ' ∈ H                   |
|                                                                               |
|  Input Contract:   |ψ(x, t0)⟩ ∈ H, Energy Surface E(ψ), Temperature T_eff      |
|  Output Contract:  |ψ(x, t0 + Δt)⟩ ∈ H conforming to Gibbs-Boltzmann measure  |
|                                                                               |
|  Strict Principle: Thermal sampling is a PHYSICAL thermodynamic property      |
|                    of the hardware substrate, NOT a digital softmax function. |
|                                                                               |
+-------------------------------------------------------------------------------+
```

### 1.1 Physical Thermodynamic Principle vs Digital Softmax
In conventional digital transformers, stochastic generation is implemented by applying an artificial mathematical softmax function over an unnormalized logit vector $\mathbf{z} \in \mathbb{R}^{V}$:

$$p_i = \frac{\exp(z_i / T)}{\sum_{j=1}^{V} \exp(z_j / T)}$$

In PhysLM, **there is no digital softmax operation**. Stochasticity originates from physical thermal fluctuations (e.g. Johnson-Nyquist noise in resistors or spontaneous emission fluctuations in optical cavities). The probability distribution over states arises naturally from thermodynamic equilibrium:

$$\boxed{P_{\text{eq}}(\psi) \propto \exp\left( -\frac{E(\psi)}{k_B T} \right)}$$

Subsequent Born-rule measurement (RFC-002) samples symbolic tokens directly from this physical thermodynamic ensemble.

---

## 2. Energy / Free-Energy Landscape

The stochastic relaxation operates over a scalar energy functional $E(\psi) \in \mathbb{R}$ defined on the continuous state $|\psi\rangle \in \mathcal{H}$:

$$\boxed{E(\psi) = E_{\text{internal}}(\psi) + E_{\text{crossbar}}(\psi) + E_{\text{Hopfield}}(\psi)}$$

where:
1. **Internal Field Energy** (from RFC-003):
   $$E_{\text{internal}}(\psi) = \int_{\Omega} \left[ \beta \left|\frac{\partial\psi}{\partial x}\right|^2 + V_{\text{trap}}(x) |\psi(x)|^2 + \frac{1}{2} g |\psi(x)|^4 \right] dx$$
2. **Associative Crossbar Energy** (from RFC-001):
   $$E_{\text{crossbar}}(\psi) = -\frac{1}{2} \int_{\Omega} \int_{\Omega} \psi^*(x) G(x, x') \psi(x') \, dx \, dx'$$
   where $G(x, x')$ is the continuous analog conductance kernel.
3. **Continuous Modern Hopfield Potential**:
   $$E_{\text{Hopfield}}(\psi) = -\frac{1}{\beta_{\text{hop}}} \ln \sum_{\mu=1}^{P} \exp\left( \beta_{\text{hop}} \text{Re}\langle \xi_\mu | \psi \rangle \right)$$
   which carves steep, well-separated attractor basins corresponding to valid syntactic and semantic concepts.

The functional gradient $\nabla_\psi E = \frac{\delta E}{\delta \psi^*(x)}$ defines the conservative drift force driving the system toward energy minima.

---

## 3. Langevin Stochastic Dynamics

The trajectory of the continuous state $|\psi(x,t)\rangle$ under thermal agitation is governed by the **overdamped complex Langevin Stochastic Differential Equation (SDE)**:

$$\boxed{d\psi(x,t) = -\mu \frac{\delta E(\psi)}{\delta \psi^*(x)} dt + \sqrt{2 \mu k_B T} \, dW_t(x)}$$

where:
- $\mu \in \mathbb{R}^+$ is the **mobility coefficient** (the inverse of physical damping / viscous friction $\gamma$, $[\mu] = [\text{force}]^{-1} [\text{velocity}]$).
- $k_B$ is the Boltzmann constant (in simulation units, $k_B \equiv 1.0$).
- $T$ is the effective thermodynamic temperature ($T \ge 0$).
- $W_t(x)$ is a complex-valued spatio-temporal Wiener process.

### Complex Wiener Process Convention:
To ensure isotropic noise in the complex plane, the Wiener increment is strictly defined as:

$$\boxed{dW_t(x) = \frac{dW_R(x, t) + i \, dW_I(x, t)}{\sqrt{2}}}$$

where $dW_R(x,t)$ and $dW_I(x,t)$ are two mutually independent real Gaussian Wiener increments satisfying:
$$\mathbb{E}[dW_R] = \mathbb{E}[dW_I] = 0$$
$$\mathbb{E}[dW_R(x,t) dW_R(x',t')] = \delta(x - x') \delta(t - t') \, dt$$
$$\mathbb{E}[dW_I(x,t) dW_I(x',t')] = \delta(x - x') \delta(t - t') \, dt$$
$$\mathbb{E}[dW_R(x,t) dW_I(x',t')] = 0$$

Consequently:
$$\mathbb{E}[dW_t^*(x) dW_t(x')] = \delta(x - x') \, dt$$

---

## 4. Fokker-Planck Formulation

While the Langevin SDE describes the individual stochastic trajectory of a single state $\psi(t)$, the ensemble probability density functional $P(\psi, t)$ over Hilbert space $\mathcal{H}$ evolves deterministically according to the functional **Fokker-Planck (Smoluchowski) Equation**:

$$\boxed{\frac{\partial P(\psi, t)}{\partial t} = \int_{\Omega} \frac{\delta}{\delta \psi(x)} \cdot \left[ \mu P(\psi, t) \frac{\delta E}{\delta \psi^*(x)} + \mu k_B T \frac{\delta P(\psi, t)}{\delta \psi^*(x)} \right] dx + \text{c.c.}}$$

### Stationary Equilibrium Solution:
At thermodynamic equilibrium ($\partial_t P = 0$), the probability current vanishes identically:

$$\mu P_{\text{eq}} \frac{\delta E}{\delta \psi^*} + \mu k_B T \frac{\delta P_{\text{eq}}}{\delta \psi^*} = 0 \implies \frac{\delta \ln P_{\text{eq}}}{\delta \psi^*} = -\frac{1}{k_B T} \frac{\delta E}{\delta \psi^*}$$

Integrating over functional space yields the unique stationary **Gibbs-Boltzmann distribution**:

$$\boxed{P_{\text{eq}}(\psi) = \frac{1}{\mathcal{Z}} \exp\left( -\frac{E(\psi)}{k_B T} \right)}$$

where the partition function $\mathcal{Z}$ normalizes the functional measure:
$$\mathcal{Z} = \int_{\mathcal{H}} \exp\left( -\frac{E(\psi)}{k_B T} \right) \mathcal{D}\psi$$

---

## 5. Fluctuation-Dissipation Relation

The drift coefficient ($\mu$) and the diffusion coefficient ($D = \mu k_B T$) are not independent tuning parameters. They are fundamentally coupled by the **Einstein-Smoluchowski Fluctuation-Dissipation Theorem**:

$$\boxed{D = \mu k_B T}$$

### Physical Implications:
1. **Thermodynamic Consistency**: Any increase in mobility $\mu$ simultaneously accelerates deterministic gradient descent AND increases thermal fluctuations.
2. **Noise Covariance Invariant**:
   For discrete numerical simulation with time-step $\Delta t$ on spatial lattice spacing $\Delta x$:
   $$\sigma_{\text{noise}}^2 = \mathbb{E}[|\eta_i|^2] = \frac{2 \mu k_B T \Delta t}{\Delta x}$$
   Implementations must scale the discrete Gaussian generator strictly by $\sqrt{\frac{2 \mu k_B T \Delta t}{\Delta x}}$ to maintain lattice-independent thermodynamic behavior.

---

## 6. Thermal Noise Model: Johnson-Nyquist & Optical Phase Noise

PhysLM specifies two physical noise models for hardware realization:

### 6.1 Johnson-Nyquist Resistor Thermal Noise (Electronics)
In analog memristive crossbars (RFC-001/RFC-005), physical thermal agitation in resistors generates a voltage fluctuation with white-noise spectral density:

$$S_V(f) = 4 k_B T R \quad [\text{V}^2 / \text{Hz}]$$

This produces an intrinsic current fluctuation entering each crossbar node:
$$\delta I_{\text{thermal}}(t) \sim \mathcal{N}\left( 0, \frac{4 k_B T G}{\Delta t} \right)$$
which maps directly to the Langevin noise term $dW_t$.

### 6.2 Amplified Spontaneous Emission (ASE) Noise (Photonics)
In nanophotonic meshes, optical amplification introduces spontaneous emission photons with spectral density:
$$S_{\text{opt}}(f) = n_{\text{sp}} (G - 1) h \nu$$
producing phase and amplitude fluctuations in the complex optical field.

---

## 7. Boltzmann Sampling & Thermally Activated Barrier Crossing

```text
+-------------------------------------------------------------------------------+
|                       PHYSICAL GENERATION PIPELINE                            |
+-------------------------------------------------------------------------------+
|                                                                               |
|   Energy Landscape E(ψ)                                                       |
|             │                                                                 |
|             ▼                                                                 |
|   Overdamped Langevin SDE:  dψ = -μ ∇E dt + √(2μ k_B T) dW_t                 |
|             │                                                                 |
|             ▼                                                                 |
|   Thermal Equilibrium Ensemble:  P(ψ) ∝ exp(-E(ψ) / k_B T)                    |
|             │                                                                 |
|             ▼                                                                 |
|   Hilbert Born-Rule Measurement:  S(c | ψ) = |⟨φ_c | ψ⟩|²                     |
|             │                                                                 |
|             ▼                                                                 |
|   Discrete Token Readout:  c ~ p(c | ψ)                                       |
|                                                                               |
+-------------------------------------------------------------------------------+
```

### 7.1 Rejection of "Thermal Tunneling" Terminology
> [!IMPORTANT]
> **Strict Scientific Terminology**:  
> In accordance with the Phase I Freeze ([`10_EXPERIMENTAL_BASELINE_FREEZE.md`](../benchmarks/10_EXPERIMENTAL_BASELINE_FREEZE.md)), transitions between distinct attractor basins under classical Langevin thermal noise must be designated as **Thermally Activated Barrier Crossing** (governed by Kramers' escape rate $\Gamma \propto \exp(-\Delta E / k_B T)$).  
> The term **tunneling** is strictly reserved for barrier penetration where $E < V_{\text{barrier}}$ via macroscopic quantum wave mechanics.

### 7.2 Relative Attractor Probability Ratios
For two distinct semantic attractors $\psi_A$ and $\psi_B$ with energy levels $E_A = E(\psi_A)$ and $E_B = E(\psi_B)$:

$$\boxed{\frac{P(\psi_A)}{P(\psi_B)} = \exp\left( -\frac{E_A - E_B}{k_B T} \right)}$$

- If $E_A < E_B$, attractor $\psi_A$ is exponentially favored.
- As $T \to 0$, $P(\psi_A) / P(\psi_B) \to \infty$ (deterministic selection of ground state).
- As $T \to \infty$, $P(\psi_A) / P(\psi_B) \to 1$ (uniform exploration).

---

## 8. Temperature Parameterization & Regimes

PhysLM parameterizes temperature via the **non-dimensional thermal scale**:

$$\boxed{\theta = \frac{k_B T}{E_0}}$$

where $E_0$ is the characteristic energy barrier between adjacent character attractors in Subsystem 3.

### Three Physical Temperature Regimes

| Regime | Range | Physical Behavior | Generative Function |
| :--- | :--- | :--- | :--- |
| **Quenched ($T \to 0$)** | $\theta < 0.05$ | Deterministic gradient descent into nearest local basin. Zero barrier crossing. | Greedy decoding, code syntax generation, exact retrieval. |
| **Ergodic (Intermediate)** | $0.05 \le \theta \le 0.50$ | Metastable exploration; thermally activated barrier crossings occur at Kramers rate. | Creative natural language generation, associative diversity. |
| **Thermalized ($T \to \infty$)** | $\theta > 1.0$ | Thermal fluctuations overwhelm potential wells; state wanders randomly in Hilbert space. | Unconstrained exploration, high-entropy hallucination, noise reset. |

---

## 9. Coupling to the Wave Dynamics Engine (RFC-003)

The Thermal Engine couples to the deterministic Wave Dynamics Engine (RFC-003) as an additive stochastic driving operator:

$$\boxed{\mathcal{D}_{\Delta t}^{T}: (\psi_t, E, T) \longmapsto \psi_{t+\Delta t}}$$

### Composite Physical Flow:
At each simulation step:
1. **Deterministic Propagation** (RFC-003):
   $$\psi_{t + \Delta t / 2} = \mathcal{D}_{\Delta t}^{\text{wave}} \psi_t$$
2. **Thermal Langevin Nudge** (RFC-004):
   $$\psi_{t + \Delta t} = \psi_{t + \Delta t / 2} - \mu \nabla E(\psi_{t + \Delta t / 2}) \Delta t + \sqrt{2 \mu k_B T \Delta t} \, \xi_t$$
   where $\xi_t \sim \mathcal{CN}(0, 1)$ is a complex standard normal vector.
3. **Unitary Renormalization** (RFC-002):
   $$\psi_{t + \Delta t} \longleftarrow \frac{\psi_{t + \Delta t}}{\|\psi_{t + \Delta t}\|}$$

---

## 10. Measurement & Sampling Interface

The final probability of emitting symbol $c \in \Sigma$ given thermal state $\psi$ is computed strictly according to the **RFC-002 Measurement Interface**:

$$S(c | \psi) = |\langle \phi_c | \psi \rangle|^2$$

$$p(c | \psi) = \frac{S(c | \psi)}{\sum_{j \in \Sigma} S(j | \psi)}$$

- At low temperature ($T \to 0$), the wave settles tightly into an attractor $|\phi_{c^*}\rangle$, producing $p(c^* | \psi) \approx 1.0$.
- At elevated temperature ($T > 0$), thermal fluctuations spread the wave across overlapping attractor basins, producing non-zero measurement probabilities across semantically related candidate characters.

---

## 11. Numerical Integration: Euler-Maruyama Method

For software simulation, the Langevin SDE is integrated using the canonical **Euler-Maruyama numerical scheme**:

$$\boxed{\psi_{n+1}(x) = \psi_n(x) - \mu \left[ \frac{\delta E}{\delta \psi^*(x)} \right]_n \Delta t + \sqrt{\frac{\mu k_B T \Delta t}{\Delta x}} \cdot \left( \xi_{R, n}(x) + i \xi_{I, n}(x) \right)}$$

where:
- $\xi_{R, n}(x), \xi_{I, n}(x) \stackrel{\text{iid}}{\sim} \mathcal{N}(0, 1)$.
- Order of convergence: strong order $\gamma = 0.5$, weak order $\beta = 1.0$.

### Numerical Stability Guard:
To prevent numerical divergence caused by finite-step discretization:
$$\Delta t \le \frac{1}{2 \mu \lambda_{\max}(\nabla^2 E)}$$
where $\lambda_{\max}$ is the maximum eigenvalue of the energy Hessian.

---

## 12. Normative Conformance Tests

Compliant implementations of the Thermal / Boltzmann Engine must satisfy all eight normative test cases:

```text
====================================================================================================
RFC-004 NORMATIVE CONFORMANCE SUITE
====================================================================================================
Test ID      Name                               Pass Condition
----------------------------------------------------------------------------------------------------
THERMAL-001  Zero-Temperature Determinism       ||ψ(t, T=0) - ψ_det(t)|| < 10^-6 for identical seeds
THERMAL-002  Zero-Mean Noise Invariant          |E[η]| < 10^-3 over 10^5 noise realizations
THERMAL-003  Fluctuation-Dissipation Variance   |Var(η) - 2μ k_B T Δt / Δx| / Var(η) < 0.05
THERMAL-004  Fokker-Planck Equilibrium Ratio    |P(A)/P(B) - exp(-ΔE / k_B T)| / (P(A)/P(B)) < 0.10
THERMAL-005  Empirical Boltzmann Match          Empirical histogram matches Gibbs distribution (R² > 0.95)
THERMAL-006  Monotonic Thermal Entropy Sweep    Entropy H(T_1) < H(T_2) whenever T_1 < T_2
THERMAL-007  Thermal Horizon Stability          Zero NaN / Inf over 10^3 steps at T_eff = 1.0
THERMAL-008  RFC-002 Interface Invariance       Post-relaxation state satisfies | ||ψ||^2 - 1.0 | < 10^-6
====================================================================================================
```

These normative test cases are automated in [`tests/test_rfc004_conformance.py`](../../tests/test_rfc004_conformance.py).
