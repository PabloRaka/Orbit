# 10 - Milestone Phase I: Experimental Baseline Freeze
## Formal Audit of Empirical Discoveries, Numerical Invariants, Substrate Models, and Open Limitations

---

## 1. Executive Summary & Freeze Declaration

This document establishes the **authoritative baseline freeze** for **Phase I: Empirical Discovery** of the Physical Language Model (PhysLM / Project Resonon).

Having completed:
$$\text{PASS-0} \longrightarrow \text{PASS-1} \longrightarrow \text{EP-01} \longrightarrow \text{EP-02} \longrightarrow \text{EP-03} \longrightarrow \text{EP-04 (Characterization)} \longrightarrow \text{TIER-4}$$

We hereby **freeze all empirical evidence** and halt the addition of speculative model features. Phase II (Architecture Consolidation) will codify these findings into strict architectural contracts (RFC-001 through RFC-005).

To prevent methodological overreach, every finding is strictly partitioned into four epistemic categories:
1. **`MEASURED`**: Directly measured and timed on host hardware (x86 CPU).
2. **`SIMULATED`**: Numerically verified inside discrete mathematical solvers (JAX/NumPy).
3. **`MODELED`**: Analytically derived from solid-state physics equations (not yet fabricated).
4. **`HYPOTHETICAL`**: Unverified conjectures, open failure modes, and long-horizon extrapolations.

---

## 2. Category 1: MEASURED (Host-Level Software Execution)

The following metrics are empirically measured through automated execution in [`tests/`](../../tests/) and [`benchmarks/`](../../benchmarks/):

| Metric / Phenomenon | Measured Value | Verification Harness | Epistemic Status |
| :--- | :--- | :--- | :--- |
| **Test Suite Pass Rate** | **35 Passed, 1 Skipped** (0 Failures) | `pytest` (36 collected items) | `MEASURED` |
| **Single-Step Generation Latency** | **$124.97\,\mu\text{s}$** ($0.125\,\text{ms}$) | Single-thread x86-64 CPU (`tier4_scaling_benchmark.py`) | `MEASURED` |
| **Autoregressive Throughput** | **$\approx 8,002\,\text{transitions/s}$** | Host CPU execution | `MEASURED` |
| **Active Memory Footprint ($M_{\text{active}}$)** | **$1,106.26\,\text{KB}$** ($\approx 1.08\,\text{MB}$) | Memory allocator audit across $N \in \{1\text{k}, 8\text{k}, 32\text{k}, 128\text{k}\}$ | `MEASURED` |
| **Memory Growth Slope ($a$)** | **$a = \mathbf{0.000000}\,\text{bytes/token}$** | Linear regression: $\max_N M - \min_N M = 0.0$ bytes | `MEASURED` |
| **Operational History Buffer ($M_{\text{history}}$)** | **$0\,\text{bytes}$** | Invariant state audit (no token storage) | `MEASURED` |
| **Held-Out Transition Top-1 Accuracy** | **$89.6\%$** | Held-out natural sentence test split (`ep04_autoregressive_scaling.py`) | `MEASURED` |
| **Held-Out Separation Margin ($M_{\text{held-out}}$)** | **$+0.1121 > 0$** | $S_{\text{tgt}} = 0.885$, $S_{\text{comp}} = 0.773$ | `MEASURED` |
| **Training Split Separation Margin ($M_{\text{train}}$)** | **$+0.1370 > 0$** | Top-1 accuracy $91.7\%$ ($48$ transitions) | `MEASURED` |
| **Exposure Bias Gap ($\Delta_{\text{bias}}$)** | **$+0.9516$** | Step-1 Teacher-Forced MSE ($0.1953$) vs Free-Running MSE ($1.1469$) | `MEASURED` |
| **Sentence Rollout Valid Character Rate (VCR)** | **$100.0\%$** | Prompt seeds (`THE C`, `THE D`, `THE S`) emitted valid syntax | `MEASURED` |
| **Sentence Rollout EOS Emission Rate** | **$100.0\%$** | Correct termination on period (`.`) delimiter | `MEASURED` |
| **Mean Sentence Trajectory Coherence ($R_\phi$)** | **$0.8874$** | Natural language rollout horizon $H \le 16$ | `MEASURED` |
| **Mean Sentence Temporal Drift ($\Delta_{\text{drift}}$)** | **$0.1126$** | Trajectory variance against teacher path | `MEASURED` |
| **Long-Horizon Error Mode A ($H=256$ Free-Flight)** | **$L(256) = 2.0217$** | Uncollapsed wave dispersion towards orthogonal limit | `MEASURED` |
| **Long-Horizon Error Mode B ($H=256$ Projective)** | **$L(256) = 1.3448$** | Born-rule projection resets phase drift ($1.50\times$ lower error) | `MEASURED` |

---

## 3. Category 2: SIMULATED (Numerical Discrete Solvers)

The following invariants are verified mathematically within discrete software simulations:

| Physical Invariant / Mechanism | Simulated Result | Solver / Environment | Epistemic Status |
| :--- | :--- | :--- | :--- |
| **Symplectic Unitary Norm Conservation** | **$|\Delta N| < 10^{-14}$** (Machine Precision) | 2nd-order Strang split-operator FFT (`src/baseline_phase0.py`) | `SIMULATED` |
| **Symplectic Energy Conservation** | **$|\Delta E / E_0| < 10^{-5}$** | Hamiltonian wave integrator over $10^4$ steps | `SIMULATED` |
| **Dyck Cavity Hierarchical Recursion** | **Depth $D=16$** balanced parenthesis | Multi-mode harmonic cavity (`src/dyck_resonator.py`) | `SIMULATED` |
| **Dyck Ground State Energy** | **$E_{\text{ground}} = 0.000000$** | Complete energy return upon parenthesis resolution | `SIMULATED` |
| **Dyck LIFO Phase Invariant** | **$\Delta \phi = 0.000000$** | Phase conjugate mode cancellation | `SIMULATED` |
| **Semantic Infilling Margin ($M$)** | **$+0.6955$** | Continuous modern Hopfield attractor (`src/associative_memory.py`) | `SIMULATED` |
| **Thermal Noise Robustness** | **$100.0\%$ Retrieval** | Infilling under Gaussian noise ($\sigma = 0.15$) | `SIMULATED` |
| **Out-Of-Distribution Rejection** | **$S_{\text{OOD}} < 0.10$** | Uncorrelated wave packets rejected | `SIMULATED` |
| **Equilibrium Propagation Convergence** | **$M_{\text{train}} = +0.2766 > 0$** | Local contrastive energy minimization without backprop | `SIMULATED` |

---

## 4. Category 3: MODELED (Analytical Physical Substrate Projections)

The following figures represent **first-principles physical models** derived from condensed matter and electrodynamics equations. They are **not measured from fabricated physical chips**:

| Parameter / Operational Property | Analog Memristive Crossbar | Nanophotonic Mesh | Underlying Physical Model / Equation | Epistemic Status |
| :--- | :--- | :--- | :--- | :--- |
| **Physical Propagation Latency** | **$10 - 50\,\text{ns}$** | **$10 - 50\,\text{ps}$** | Memristor: RC time constant $\tau = R_{\text{on}} C_{\text{cell}}$<br>Photonics: Optical flight time $\tau = n_{\text{eff}} L / c$ | `MODELED` |
| **Operational DRAM Footprint** | **$0\,\text{bytes}$** | **$0\,\text{bytes}$** | In-situ state storage: Analog conductance $G_{ij}$ & optical waveguides | `MODELED` |
| **Off-Chip Memory Bus Bandwidth** | **$0\,\text{GB/s}$** | **$0\,\text{GB/s}$** | Compute-in-Memory bypasses Von Neumann bus | `MODELED` |
| **Energy Consumption per MAC** | **$\approx 1.2\,\text{pJ}$** | **$\approx 50\,\text{fJ}$** | Memristor: Joule heating $E = V^2 G \Delta t$<br>Photonics: Laser diode power + photodetector capacitive limit | `MODELED` |
| **Context Complexity Scaling** | **$\mathcal{O}(1)$ Invariant** | **$\mathcal{O}(1)$ Invariant** | Fixed physical volume independent of processed context | `MODELED` |

### Digital Baseline Reference (Theoretical & Measured):
- **Llama-3-8B GQA HBM Bandwidth Requirement**:
  $$\text{Bandwidth}(N=128\text{k}) = 16.384\,\text{GB} \times 30\,\text{tokens/s} = \mathbf{480.0\,\text{GB/s}}$$
  This demonstrates that digital Transformers are fundamentally bound by the memory bandwidth wall, an artifact completely absent from in-situ physical substrates.

---

## 5. Category 4: HYPOTHETICAL (Open Limitations & Unproven Claims)

The following items are **explicitly unproven** and must not be claimed as verified until further empirical research is conducted:

1. **Infinite Asymptotic Context Invariance ($N \to \infty$)**:
   - *Status*: `HYPOTHETICAL`.
   - *Reason*: While active memory was proved invariant across $N \in \{1\text{k}, 8\text{k}, 32\text{k}, 128\text{k}\}$ ($a=0.0$), real physical media suffer from finite dynamic range, thermal drift ($dG/dt \neq 0$), and spectral leakage when integrating infinite continuous sequences.
2. **Open-Domain Natural Language Generation**:
   - *Status*: `HYPOTHETICAL`.
   - *Reason*: Verified vocabulary is bounded ($|\Sigma| \le 50$ characters, short sentences). Vocabulary scaling to $100\text{k}$ semantic concepts without cross-talk interference has not been demonstrated.
3. **Unassisted Pure Analog Long-Horizon Stability ($L(H) \to 0$ without Measurement)**:
   - *Status*: `HYPOTHETICAL / REFUTED IN MODE A`.
   - *Reason*: At $H=256$, uncollapsed free-flight waves disperse ($L(256)=2.0217$, $R_\phi=0.2769$). Bounded state memory does not automatically imply bounded dynamical stability. Whether a continuous Hamiltonian system can stay on the language manifold without discrete projective measurement remains an open theoretical problem.
4. **Multi-Layer Physical Equilibrium Propagation**:
   - *Status*: `HYPOTHETICAL`.
   - *Reason*: EP learning has only been verified on a single-layer crossbar ($N_{\text{grid}} \times N_{\text{grid}}$). Multi-layer continuous wave EqProp with non-linear optical phase transitions has not been simulated.
5. **Physical Fabrication Imperfections**:
   - *Status*: `HYPOTHETICAL`.
   - *Reason*: Simulations assume ideal floating-point precision or uniform memristor grids. Real devices have device-to-device variability ($\sigma_G / \mu_G \approx 5-10\%$), optical phase drift, and ADC/DAC boundary bottlenecks.

---

## 6. The Master Baseline Freeze Matrix

```text
====================================================================================================
PROJECT RESONON / PHYSLM: PHASE I EXPERIMENTAL BASELINE FREEZE MATRIX
====================================================================================================
Subsystem / Finding                    Metric / Value              Status      Harness / Source
----------------------------------------------------------------------------------------------------
Automated Unit & Regr Tests            35 Passed, 1 Skipped        MEASURED    pytest (36 items)
Single-Step Generation Latency (CPU)   124.97 µs (8,002 tok/s)     MEASURED    tier4_scaling_benchmark
Operational Memory Scaling M(N)        1,106.26 KB (Slope a=0.0)   MEASURED    tier4_scaling_benchmark
Hidden Operational History Buffer      0 bytes                     MEASURED    tier4_scaling_benchmark
Ingestion / Reading Complexity         C(N) ~ N^1.0149 (Linear)    MEASURED    tier4_scaling_benchmark
Autoregressive Step Complexity         C(N) ~ N^0.0000 (O(1))      MEASURED    tier4_scaling_benchmark
Held-Out Transition Margin (M_heldout) +0.1121 (Acc: 89.6%)        MEASURED    ep04_autoregressive
Sentence Rollout VCR / EOS             100.0% / 100.0%             MEASURED    ep04_autoregressive
Long-Horizon Error H=256 (Mode A)      L(256) = 2.0217             MEASURED    ep04_autoregressive
Long-Horizon Error H=256 (Mode B)      L(256) = 1.3448 (VCR: 100%) MEASURED    ep04_autoregressive
----------------------------------------------------------------------------------------------------
Symplectic Unitary Norm Invariant      |ΔN| < 10^-14               SIMULATED   baseline_phase0.py
Symplectic Energy Invariant            |ΔE / E0| < 10^-5           SIMULATED   baseline_phase0.py
Dyck Cavity Balanced Recursion         Depth D=16 (E=0.000000)     SIMULATED   ep02_dyck_resonance.py
Semantic Associative Infilling         Margin M = +0.6955          SIMULATED   ep03_semantic_infilling
----------------------------------------------------------------------------------------------------
Memristive Crossbar Step Latency       10 - 50 ns (RC limit)       MODELED     RC physical equation
Memristive Crossbar Energy / MAC       ~ 1.2 pJ                    MODELED     V^2 G Δt Joule model
Nanophotonic Mesh Step Latency         10 - 50 ps (Flight time)    MODELED     n_eff L / c optics
Nanophotonic Mesh Energy / Trans       ~ 50 fJ                     MODELED     Detector capacitive limit
Compute-In-Memory DRAM Bandwidth Req   0 GB/s                      MODELED     In-situ physical state
----------------------------------------------------------------------------------------------------
Infinite Sequence Dynamic Range        Unknown at N -> ∞           HYPOTHETICAL Finite media boundary
Open-Domain Vocab Scaling (>100k)      Unknown at scale            HYPOTHETICAL Cross-talk boundary
Autonomous Long-Horizon Free-Flight    Refuted in Mode A           HYPOTHETICAL Requires measurement
Deep Multi-Layer Continuous EqProp     Unsimulated                 HYPOTHETICAL Multi-layer physical
Hardware Noise / Variability Tolerance Unsimulated on wafer        HYPOTHETICAL Fabrication boundary
====================================================================================================
```

---

## 7. Milestone Approval & Phase Transition

Phase I (Empirical Discovery) is hereby formally **CLOSED AND FROZEN**.

The project transitions directly into **Phase II (Architecture Consolidation)**:
- **RFC-001**: PhysLM Core Architecture (Paradigm & Subsystem Boundaries)
- **RFC-002**: Continuous Hilbert State Specification
- **RFC-003**: Wave Dynamics Engine Specification
- **RFC-004**: Thermal / Boltzmann Engine Specification
- **RFC-005**: Physical Prototype Specification
