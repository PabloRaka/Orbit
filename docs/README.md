# Physical Language Model (PhysLM / Project Resonon)
## Native Physics-Based Architecture Documentation Suite

This documentation suite establishes the foundational architecture for building a Language Model natively on physical principles, bypassing the digital Transformer paradigm (Matrix Multiplications, Self-Attention, Discrete Tokenization, and Global Backpropagation).

---

## Architectural Backbone

The architecture is founded on a **Triadic Physical Engine**:
1. **Quantum Hilbert Space for Representation:** Multi-layered continuous phase/amplitude encoding replacing discrete token IDs.
2. **Non-linear Wave Dynamics for Evolution:** Physical wave dispersion and cavity resonance replacing $O(N^2)$ Self-Attention mechanisms.
3. **Thermodynamics & Dissipative Mechanics for Optimization & Sampling:** Free energy minimization and physical Johnson-Nyquist thermal noise replacing digital softmax and backpropagation.

---

## Documentation Structure

```
docs/
├── README.md                      # Master Navigation & Document Lifecycle
├── backbone/                      # Foundational Architecture (Ground Truth)
│   ├── README.md                  # Backbone Navigation Index
│   ├── 01_VISION_AND_GRAND_ARCHITECTURE.md
│   ├── 02_MATHEMATICAL_AND_PHYSICAL_FORMULATION.md
│   ├── 03_SOFTWARE_SIMULATION_SPECIFICATION.md
│   └── 04_HARDWARE_ROADMAP_AND_MAPPING.md
├── rfcs/                          # Modular RFC Subsystem Specifications
│   ├── README.md                  # RFC Registry & Lifecycle
│   ├── RFC-001_TOKENLESS_SPECTRAL_TRANSDUCER.md
│   ├── RFC-002_SYMPLECTIC_HAMILTONIAN_INTEGRATORS.md
│   ├── RFC-003_EQUILIBRIUM_PROPAGATION_MEMRISTIVE_CROSSBAR.md
│   └── RFC-004_THERMAL_NOISE_BOLTZMANN_SAMPLING.md
└── benchmarks/                    # Simulation Results & Physical Verification
    ├── 01_PHYSICAL_AND_NUMERICAL_BENCHMARK_SUITE.md
    ├── 02_PHASE0_EMPIRICAL_BASELINE_RESULTS.md
    ├── 03_TIER4_COMPARATIVE_SCALING_RESULTS.md
    ├── 04_AUTOREGRESSIVE_TRAINING_RESULTS.md
    ├── 05_EP01_LEARNING_DYNAMICS_VALIDATION.md
    ├── 06_EP02_DYCK_GRAMMAR_RESONANCE.md
    ├── 07_EP03_SEMANTIC_ASSOCIATIVE_INFILLING.md
    ├── 08_EP04_AUTOREGRESSIVE_SCALING.md
    ├── 09_TIER4_COMPARATIVE_SCALING_SUITE.md
    └── 10_EXPERIMENTAL_BASELINE_FREEZE.md
```

---

## Architectural Backbone

The foundational architecture is documented in [`docs/backbone/`](backbone/README.md):

| Document | Title | Description |
| :--- | :--- | :--- |
| **[Doc 01](backbone/01_VISION_AND_GRAND_ARCHITECTURE.md)** | **Vision & Grand Architecture** | High-level paradigm shift, why Transformers are silicon artifacts, system topology, and the 3-tiered learning hierarchy. |
| **[Doc 02](backbone/02_MATHEMATICAL_AND_PHYSICAL_FORMULATION.md)** | **Mathematical & Physical Formulation** | Rigorous mathematical equations, Hilbert state representations, Non-linear Schrödinger/Ginzburg-Landau dynamics, and action principles. |
| **[Doc 03](backbone/03_SOFTWARE_SIMULATION_SPECIFICATION.md)** | **Software Simulation Specification** | Computational architecture using JAX and Diffrax, numerical symplectic integrators, and PyTorch compatibility bridge. |
| **[Doc 04](backbone/04_HARDWARE_ROADMAP_AND_MAPPING.md)** | **Hardware Mapping & Phased Roadmap** | Phased trajectory from software ODE/SDE solvers to Analog CMOS, Mid-Term Tri-Substrate Hybrids, and Long-Term All-Photonic architectures. |

---

## Subsystem Specifications (RFCs)

Documented in [`docs/rfcs/`](rfcs/README.md):

| RFC | Title | Subsystem Area |
| :--- | :--- | :--- |
| **[RFC-001](rfcs/RFC-001_PHYSLM_CORE_ARCHITECTURE.md)** | **PhysLM Core Architecture & System Paradigm** | Overall Paradigm & Boundaries (Ratified Contract) |
| **[RFC-002](rfcs/RFC-002_CONTINUOUS_HILBERT_STATE.md)** | **Continuous Hilbert State Specification** | Hilbert Space, Basis, Metrics, Measurement (Ratified Contract) |
| **[RFC-003](rfcs/RFC-003_WAVE_DYNAMICS_ENGINE.md)** | **Wave Dynamics Engine Specification** | Hamiltonian Flow, Cavity, Solvers (Ratified Contract) |
| **[RFC-004](rfcs/RFC-004_THERMAL_BOLTZMANN_ENGINE.md)** | **Thermal / Boltzmann Engine Specification** | Langevin SDE, Hardware Thermal Noise (Ratified Contract) |
| **[RFC-005](rfcs/RFC-005_PHYSICAL_PROTOTYPE_SPECIFICATION.md)** | **Physical Prototype Specification** | Hardware Mapping (Photonics/Crossbar) (Ratified Contract) |

---

## Verification & Benchmarks

Documented in [`docs/benchmarks/`](benchmarks/01_PHYSICAL_AND_NUMERICAL_BENCHMARK_SUITE.md):

| Report | Title | Description |
| :--- | :--- | :--- |
| **[Benchmark 01](benchmarks/01_PHYSICAL_AND_NUMERICAL_BENCHMARK_SUITE.md)** | **4-Tier Benchmark Specification** | Test protocols for Physics, Mechanisms, Language, and Scaling. |
| **[Results 02](benchmarks/02_PHASE0_EMPIRICAL_BASELINE_RESULTS.md)** | **Phase 0 Empirical Baseline Results** | Machine-precision norm & energy calibration data. |
| **[Results 03](benchmarks/03_TIER4_COMPARATIVE_SCALING_RESULTS.md)** | **Tier 4 Comparative Scaling Results** | 29,491x memory footprint advantage over NanoGPT at 32k context. |
| **[Results 04](benchmarks/04_AUTOREGRESSIVE_TRAINING_RESULTS.md)** | **Autoregressive Training & Generation Results** | Empirical sequence learning & Boltzmann sampling without backprop. |
| **[Validation 05](benchmarks/05_EP01_LEARNING_DYNAMICS_VALIDATION.md)** | **Milestone EP-01: Learning Dynamics Validation** | Resolving attractor collapse; verified $M > 0$ on synthetic cycles. |
| **[Validation 06](benchmarks/06_EP02_DYCK_GRAMMAR_RESONANCE.md)** | **Milestone EP-02: Dyck Cavity Resonance** | Stackless multi-mode recursion up to $D=16$; LIFO phase & energy invariants verified. |
| **[Validation 07](benchmarks/07_EP03_SEMANTIC_ASSOCIATIVE_INFILLING.md)** | **Milestone EP-03: Semantic Associative Infilling** | Continuous modern Hopfield landscape, $M = +0.6955$, noise robustness, and gated Dyck coupling. |
| **[Validation 08](benchmarks/08_EP04_AUTOREGRESSIVE_SCALING.md)** | **Milestone EP-04: Autoregressive Scaling** | Next-wave prediction, $M_{\text{held-out}} = +0.1121$, Free-Flight vs Projective Restoration up to $H=256$. Characterization milestone (revealed long-horizon analog phase dispersion). |
| **[Benchmark 09](benchmarks/09_TIER4_COMPARATIVE_SCALING_SUITE.md)** | **Tier-4 Comparative Scaling Suite** | Rigorous comparison vs NanoGPT, Llama-3-8B, and Mamba; $M_{\text{active}}(N) = \mathcal{O}(1)$ proven ($a=0.0$), $C_{\text{ingest}} \sim \mathcal{O}(N)$, measured vs modeled substrates, and dynamical stability frontier. |
| **[Baseline Freeze 10](benchmarks/10_EXPERIMENTAL_BASELINE_FREEZE.md)** | **Phase I Experimental Baseline Freeze** | Formal audit separating Measured, Simulated, Modeled, and Hypothetical claims across Phase I. |


---

## Phase III — Physical Substrate Validation

Operational documentation for Phase III execution:

| Document | Title | Description |
| :--- | :--- | :--- |
| **[Phase III Plan](PHASE_III_EXECUTION_PLAN.md)** | **Phase III Execution Plan** | Operational 5-gate protocol (P0 Component Characterization, P1 Wave Fidelity, P2 Dyck Resonance, P3 Boltzmann Sampling, P4 Attractor Demonstrator) with Hardware-in-the-Loop calibration. |

---

## Architectural Integrity

All implementations in `src/` are tested and validated against the mathematical formulations and benchmarks documented in this suite. The documentation in `docs/` serves as the sole ground-truth specification for the project.

