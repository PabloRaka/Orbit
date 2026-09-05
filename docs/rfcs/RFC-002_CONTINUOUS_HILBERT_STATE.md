# RFC-002: Continuous Hilbert State Specification
## Phase II Architectural Contract: State Space, Harmonic Basis, Metric, and Measurement Interface

* **Status:** `RATIFIED ARCHITECTURAL CONTRACT` (Phase II Architecture Consolidation)
* **Author:** Project Resonon / PhysLM Core Architecture Group
* **Scope:** Mathematical Definition of State Space, Canonical Dual-Harmonic Basis, Inner Product, Spatial Encoding, Normalization Invariants, Measurement Operators, and Conformance Test Suite
* **Parent Architecture:** [RFC-001: PhysLM Core Architecture](RFC-001_PHYSLM_CORE_ARCHITECTURE.md)
* **Empirical Ground Truth:** [`docs/benchmarks/10_EXPERIMENTAL_BASELINE_FREEZE.md`](../benchmarks/10_EXPERIMENTAL_BASELINE_FREEZE.md)

---

## 1. State Definition

The fundamental operational state of PhysLM is a continuous complex-valued wavefield:

$$\psi(x, t) \in \mathbb{C}, \quad x \in \Omega = [-W/2, W/2]$$

where:
- $\Omega \subset \mathbb{R}$ is the 1D spatial containment domain of width $W$.
- $t \in \mathbb{R}^+$ is the continuous physical evolution parameter.
- The state space is the infinite-dimensional complex Hilbert space of square-integrable functions:

$$\boxed{|\psi\rangle \in \mathcal{H} = L^2(\Omega, \mathbb{C})}$$

### Architectural Separation:
Unlike digital language models where the hidden state is a discrete sequence of floating-point token embedding vectors $\mathbf{h} \in \mathbb{R}^{N \times d}$, in PhysLM the state is **natively continuous in space and amplitude**. Discrete vectors appear exclusively when numerical simulation lattices discretize $\Omega$ into $N_{\text{grid}}$ spatial samples.

---

## 2. Hilbert Space & Inner Product

### 2.1 Canonical Inner Product
For any two state vectors $|\phi\rangle, |\psi\rangle \in \mathcal{H}$, the canonical inner product is strictly defined as:

$$\boxed{\langle \phi | \psi \rangle = \int_{\Omega} \phi^*(x) \psi(x) \, dx}$$

where $\phi^*(x)$ denotes the complex conjugate of $\phi(x)$.

### 2.2 Numerical Lattice Quadrature
In discrete simulation across $N_{\text{grid}}$ lattice coordinates with uniform spacing $\Delta x = x_{i+1} - x_i$:

$$\langle \phi | \psi \rangle \approx \sum_{i=0}^{N_{\text{grid}}-1} \phi_i^* \psi_i \, \Delta x$$

> [!IMPORTANT]
> **Single Canonical Definition**: This inner product definition is the sole standard across all subsystems. Transducers, associative memory attractors, wave dynamics integrators, and projective measurement decoders must compute overlaps strictly via this integral quadrature.

### 2.3 Induced Norm & Metric
The induced $L^2$ norm is:

$$\|\psi\| = \sqrt{\langle \psi | \psi \rangle} = \left( \int_{\Omega} |\psi(x)|^2 \, dx \right)^{1/2}$$

The metric distance between two physical states is:

$$d_{\mathcal{H}}(\phi, \psi) = \|\phi - \psi\| = \sqrt{\langle \phi - \psi | \phi - \psi \rangle}$$

---

## 3. Continuous Character Basis

The symbolic vocabulary of language is represented by a set of normalized physical wave basis states:

$$\Phi = \{ |\phi_c\rangle : c \in \Sigma \}$$

where $\Sigma$ is the active character alphabet (e.g. printable ASCII characters $[32..126]$, $|\Sigma| = 95$).

### 3.1 Canonical Dual-Harmonic Gabor Parameterization
Each character $c \in \Sigma$ with alphabetical index $\text{idx}(c) \in \{0, \dots, |\Sigma|-1\}$ is assigned a unique pair of low and high formant frequencies $(k_{1, c}, k_{2, c})$:

$$\boxed{\phi_c(x) = A_c \exp\left( -\frac{x^2}{2 \sigma^2} \right) \cdot \frac{1}{2} \left[ \exp(i k_{1, c} x) + \exp(i k_{2, c} x) \right]}$$

where:
1. **Fundamental Wavenumber Step**:
   $$\Delta k = \frac{2\pi}{W_{\text{char}}}$$
   where $W_{\text{char}}$ is the spatial window width of a single character cell.
2. **Formant Quantization**:
   $$m = \text{idx}(c) \pmod{10}, \quad n = \text{idx}(c) // 10$$
   $$k_{1, c} = (1 + m) \cdot \Delta k \quad (\text{Low-frequency formant})$$
   $$k_{2, c} = (12 + n) \cdot \Delta k \quad (\text{High-frequency formant})$$
3. **Gaussian Envelope Width**: $\sigma$ controls spectral bandwidth and prevents high-frequency Gibbs ringing.
4. **Unitary Normalization Constant**:
   $$A_c = \left( \int_{\Omega} \exp\left(-\frac{x^2}{\sigma^2}\right) \cdot \frac{1}{4} \left| e^{i k_{1, c} x} + e^{i k_{2, c} x} \right|^2 dx \right)^{-1/2}$$
   ensuring $\|\phi_c\| = 1.0$.

### 3.2 Orthogonality & Overlap Structure
Due to the dual-harmonic frequency separation, basis states satisfy:
- **Self-Overlap**: $\langle \phi_c | \phi_c \rangle = 1.0$.
- **Cross-Character Discrimination**: For distinct characters $c \ne j$:
  $$|\langle \phi_j | \phi_c \rangle| < \epsilon_{\text{cross}} \ll 1.0$$
  Empirically, the separation margin satisfies $M = 1.0 - \max_{j \ne c} |\langle \phi_j | \phi_c \rangle| > 0.20$.

---

## 4. Spatial Position & Sequence Superposition

### 4.1 Coordinate Mapping
A sequence of characters $c_0, c_1, \dots, c_{K-1}$ is positioned along the continuous spatial axis $\Omega$ at equidistant centers:

$$x_j = x_{\text{start}} + j \cdot \Delta x_{\text{char}}, \quad j \in \{0, \dots, K-1\}$$

where $\Delta x_{\text{char}} \ge 3 \sigma$ ensures minimal inter-symbol spatial cross-talk.

### 4.2 Sequence Superposition Wavefield
The total initial state representing a sequence is the continuous superposition:

$$\boxed{\psi(x, 0) = \frac{1}{\sqrt{\mathcal{N}}} \sum_{j=0}^{K-1} a_j \phi_{c_j}(x - x_j)}$$

where:
- $a_j \in \mathbb{C}$ are complex sequence amplitudes (default $a_j = 1.0$).
- $\mathcal{N}$ is the global normalization factor:
  $$\mathcal{N} = \int_{\Omega} \left| \sum_{j=0}^{K-1} a_j \phi_{c_j}(x - x_j) \right|^2 dx$$

> [!NOTE]
> **No Discrete Positional Embeddings**: Positional order is encoded directly by the continuous spatial coordinate $x$. Shifting a character in the sequence corresponds to a physical translation group action $\mathcal{T}_{\Delta x} \psi(x) = \psi(x - \Delta x)$ on the underlying field.

---

## 5. Normalization & Invariants

Every valid physical state $|\psi\rangle \in \mathcal{H}$ must satisfy the **Unitary Normalization Invariant**:

$$\boxed{\|\psi\|^2 = \langle \psi | \psi \rangle = \int_{\Omega} |\psi(x)|^2 \, dx = 1.0}$$

### 5.1 Interface Tolerance
In software simulation, numerical discretization and finite integration grid approximations introduce small floating-point errors. The normative compliance condition is:

$$\boxed{|\|\psi\|^2 - 1.0| < 10^{-6}}$$

### 5.2 Dynamic Invariance
Under closed Hamiltonian wave dynamics (Subsystem 2 / RFC-003):

$$\frac{d}{dt} \langle \psi(t) | \psi(t) \rangle = 0$$

Empirically verified in Phase-0 symplectic FFT integrators to machine precision: $|\Delta N| < 10^{-14}$.

---

## 6. State Composition / Superposition vs Semantic State

To prevent conceptual ambiguity, PhysLM defines superposition strictly as linear combination in the Hilbert space $\mathcal{H}$:

$$|\psi\rangle = \sum_k c_k |\psi_k\rangle, \quad c_k \in \mathbb{C}$$

We explicitly distinguish four distinct physical properties of composite states:
1. **Amplitude Weighting ($|c_k|^2$)**: Represents the statistical or thermodynamic occupancy of state $|\psi_k\rangle$.
2. **Spatial Overlap ($\int \phi_1^*(x)\phi_2(x)dx$)**: Measures geometric co-location of wave packets in physical space $\Omega$.
3. **Relative Phase ($\Delta \theta = \arg(c_1) - \arg(c_2)$)**: Determines constructive vs destructive interference during wave propagation.
4. **Semantic Association**: A property not of the isolated wavefunction, but of the **attractor energy landscape** $E(\psi; G)$ in Subsystem 3 (RFC-001/RFC-003).

---

## 7. Semantic Projection

The degree to which a wavefield $|\psi\rangle$ contains a specific character or concept $c$ at position $x_j$ is given by the orthogonal projection operator:

$$\mathcal{P}_{j, c} = |\phi_{j, c}\rangle \langle \phi_{j, c}|$$

where $|\phi_{j, c}\rangle = \phi_c(x - x_j)$.

### 7.1 Complex Projection Amplitude
$$\alpha(c | \psi, x_j) = \langle \phi_{j, c} | \psi \rangle \in \mathbb{C}$$

### 7.2 Phase Coherence Metric ($R_\phi$)
The phase alignment of the projected state against the canonical basis is:

$$R_\phi = \frac{|\langle \phi_{j, c} | \psi \rangle|}{\|\phi_{j, c}\| \|\psi_{j}\|}$$

### 7.3 Manifold Distance ($\Delta_{\text{basis}}$)
The orthogonal distance from the continuous state to the valid symbolic manifold is:

$$\Delta_{\text{basis}} = 1.0 - |\langle \phi_{\hat{c}} | \psi \rangle|$$

where $\hat{c} = \arg\max_{c} |\langle \phi_c | \psi \rangle|$.

---

## 8. Measurement Interface (Complex Projection & Born-Rule Sampling)

Readout from the continuous wavefield into discrete symbolic tokens is governed by the measurement operator:

$$\boxed{S(c | \psi, x_j) = |\langle \phi_{j, c} | \psi \rangle|^2}$$

### 8.1 Deterministic Decoding (ArgMax Readout)
For greedy or deterministic decoding at spatial center $x_j$:

$$\hat{c} = \arg\max_{c \in \Sigma} |\langle \phi_{j, c} | \psi \rangle|$$

### 8.2 Born-Rule-Inspired Stochastic Sampling
For stochastic or temperature-controlled generation:

$$\boxed{p(c | \psi, x_j) = \frac{|\langle \phi_{j, c} | \psi \rangle|^2}{\sum_{j' \in \Sigma} |\langle \phi_{j', c} | \psi \rangle|^2}}$$

Satisfying:
$$\sum_{c \in \Sigma} p(c | \psi, x_j) = 1.0 \quad \text{unconditionally.}$$

> [!NOTE]
> **Methodological Terminology**: In accordance with the project baseline freeze, this operation is formally designated as **complex Hilbert-space projection and Born-rule-inspired sampling**. It provides the mathematical interface for optoelectronic detector readouts in physical substrates.

---

## 9. Noise & Perturbation Model

RFC-002 defines the standard kinematic perturbation interface for states entering dissipative or thermal relaxation:

$$\psi'(x) = \frac{\psi(x) + \eta(x)}{\|\psi + \eta\|}$$

where $\eta(x)$ is a complex Gaussian perturbation field:

$$\eta(x) = \eta_R(x) + i \eta_I(x), \quad \eta_R, \eta_I \sim \mathcal{N}\left(0, \frac{\sigma_\eta^2}{2}\right)$$

### Scope Boundary:
RFC-002 specifies only the static perturbation contract. The continuous time-evolution of thermal noise via Langevin SDEs:
$$d\psi = -\nabla E(\psi) dt + \sqrt{2 k_B T} dW_t$$
is strictly quarantined within **RFC-004 (Thermal / Boltzmann Engine)**.

---

## 10. Numerical Representation

### 10.1 Standard Lattice Parameters

| Parameter | High-Resolution Baseline | Compact Autoregressive Baseline | Physical Description |
| :--- | :--- | :--- | :--- |
| **Grid Points ($N_{\text{grid}}$)** | $1024$ | $256$ | Number of discrete spatial samples |
| **Domain Bounds ($\Omega$)** | $[-20.0, +20.0]$ | $[-10.0, +10.0]$ | Spatial coordinates in arbitrary physical units |
| **Spatial Resolution ($\Delta x$)** | $0.03906$ | $0.078125$ | Lattice spacing ($W / N_{\text{grid}}$) |
| **Data Type** | `complex64` / `complex128` | `complex64` | Complex floating-point storage |
| **Character Spacing ($\Delta x_{\text{char}}$)** | $1.5$ | $1.5$ | Center-to-center distance between symbols |
| **Envelope Width ($\sigma$)** | $0.4$ | $0.4$ | Gaussian wave packet spread |
| **Window Width ($W_{\text{char}}$)** | $1.4$ | $1.4$ | Formant orthogonality period |

---

## 11. Normative Conformance Tests

Any compliant PhysLM implementation of the Hilbert State Subsystem **must pass** all six normative test cases defined below:

```text
====================================================================================================
RFC-002 NORMATIVE CONFORMANCE SUITE
====================================================================================================
Test ID             Name                              Pass Condition
----------------------------------------------------------------------------------------------------
HilbertState-001    Self-Norm Invariant               | ||ψ||^2 - 1.0 | < 10^-6 for any encoded state
HilbertState-002    Basis Dominance Overlap           |<φ_c | φ_c>| > |<φ_j | φ_c>| + 0.15  ∀ j ≠ c
HilbertState-003    Superposition Normalization       ||ψ_composite|| = 1.0 ± 10^-6
HilbertState-004    Spatial Translation Invariance    ||T_Δx ψ|| = ||ψ|| ± 10^-6
HilbertState-005    Born Probability Unit Sum         | Σ_c p(c | ψ) - 1.0 | < 10^-6
HilbertState-006    Encode-to-Project Round Trip      argmax_c |<φ_c | ψ_target>| == c_target  (100%)
====================================================================================================
```

These normative test cases are implemented and automated in [`tests/test_rfc002_conformance.py`](../../tests/test_rfc002_conformance.py).
