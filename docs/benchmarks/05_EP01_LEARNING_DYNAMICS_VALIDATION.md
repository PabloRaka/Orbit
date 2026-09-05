# 05 - Milestone EP-01: Learning Dynamics & Attractor Landscape Validation
## Resolving Attractor Collapse via Separation Margin Optimization (M > 0)

---

## 1. Scientific Motivation & Failure Mode Diagnosis

In initial natural-language autoregressive training experiments ([Benchmark 04](04_AUTOREGRESSIVE_TRAINING_RESULTS.md)), the system exhibited a classic neuromorphic failure mode: **attractor/mode collapse**.
* While the mean-squared loss decreased ($0.0747 \to 0.0673$), greedy generative rollout settled into a single repetitive character (`"PSSSSSSSSSSSSSS"`).
* **Diagnosis**: The network developed a dominant fixed-point attractor basin. Because the energy landscape failed to carve distinct attractor wells for individual transitions, trajectories from any seed collapsed into the lowest global minimum.

To build an empirically grounded foundation before scaling to natural language, **Milestone EP-01** isolates the learning dynamics onto minimal deterministic synthetic transition cycles.

---

## 2. Core Quantitative Metric: The Separation Margin ($M$)

For any target transition $c_{\text{context}} \to c_{\text{target}}$:
1. The predicted continuous wave $\psi_{\text{pred}}(x)$ is projected onto the normalized Hilbert-space basis probes:
   $$S_j = |\langle \phi_j | \psi_{\text{pred}} \rangle|$$
2. The target overlap is:
   $$S_{\text{target}} = S_{c_{\text{target}}}$$
3. The competitor maximum overlap is:
   $$S_{\text{competitor}} = \max_{j \neq c_{\text{target}}} S_j$$
4. The **Separation Margin** is defined as:
   $$\boxed{M = S_{\text{target}} - S_{\text{competitor}}}$$

### Falsifiable Success Gate
$$\boxed{M > 0 \quad \text{consistently across all transitions}}$$
A positive margin $M > 0$ guarantees that the target attractor basin is strictly deeper than all competing attractors, ensuring deterministic and non-collapsed sequence generation.

---

## 3. Empirical Results: Task 1 & Task 2

Benchmark implementation: [`benchmarks/ep01_learning_dynamics.py`](../../benchmarks/ep01_learning_dynamics.py).

### 3.1 Task 1: 1-gram Permutation Cycle ($A \to B \to C \to D \to A$)
* **Context Window**: $K = 1$
* **Grid**: $N = 128$, $\eta = 0.06$, $\beta = 0.35$

#### Convergence Trajectory Across Epochs
| Epoch | MSE Loss | $E_{\text{free}}$ | $E_{\text{nudge}}$ | $\|\Delta W\|$ | $S_{\text{target}}$ | $S_{\text{competitor}}$ | Margin $M$ | Accuracy |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **1** | $0.06720$ | $-1.901$ | $-1.265$ | $2.4908$ | $0.876$ | $0.807$ | **$+0.0685$** | $75.0\%$ |
| **2** | $0.02354$ | $-4.988$ | $-4.765$ | $1.4874$ | $0.919$ | $0.785$ | **$+0.1347$** | $100.0\%$ |
| **5** | $0.00369$ | $-5.684$ | $-5.649$ | $0.5729$ | $0.974$ | $0.753$ | **$+0.2212$** | $100.0\%$ |
| **10** | $0.00041$ | $-6.246$ | $-6.242$ | $0.1878$ | $0.995$ | $0.732$ | **$+0.2632$** | $100.0\%$ |
| **15** | $0.00006$ | $-6.348$ | $-6.348$ | $0.0700$ | $0.998$ | $0.727$ | **$+0.2714$** | $100.0\%$ |
| **20** | $0.00001$ | $-6.380$ | $-6.380$ | $0.0256$ | $0.999$ | $0.725$ | **$+0.2747$** | $100.0\%$ |
| **25** | $\mathbf{0.00000}$ | $-6.393$ | $-6.393$ | $0.0094$ | $\mathbf{1.000}$ | $0.724$ | $\mathbf{+0.2759}$ | $\mathbf{100.0\%}$ |

#### Final Transition Breakdown
* $A \to B$: $S_{\text{target}} = 1.000$, $S_{\text{competitor}} = 0.724 \implies M = \mathbf{+0.2761}$ `[PASSED M > 0]`
* $B \to C$: $S_{\text{target}} = 1.000$, $S_{\text{competitor}} = 0.724 \implies M = \mathbf{+0.2754}$ `[PASSED M > 0]`
* $C \to D$: $S_{\text{target}} = 0.999$, $S_{\text{competitor}} = 0.724 \implies M = \mathbf{+0.2756}$ `[PASSED M > 0]`
* $D \to A$: $S_{\text{target}} = 1.000$, $S_{\text{competitor}} = 0.723 \implies M = \mathbf{+0.2765}$ `[PASSED M > 0]`

#### Generative Rollout
$$\mathbf{A \to B \to C \to D \to A \to B \to C \to D \to A}$$
*(Zero attractor collapse; perfect cyclic autoregressive generation).*

---

### 3.2 Task 2: 2-gram Context Permutation Cycle ($AB \to C, BC \to D, CD \to A, DA \to B$)
* **Context Window**: $K = 2$
* **Grid**: $N = 128$, $\eta = 0.06$, $\beta = 0.35$

#### Convergence Trajectory Across Epochs
| Epoch | MSE Loss | $E_{\text{free}}$ | $E_{\text{nudge}}$ | $\|\Delta W\|$ | $S_{\text{target}}$ | $S_{\text{competitor}}$ | Margin $M$ | Accuracy |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **1** | $0.07211$ | $-1.237$ | $-0.554$ | $2.5853$ | $0.879$ | $0.789$ | **$+0.0895$** | $100.0\%$ |
| **2** | $0.01480$ | $-4.942$ | $-4.802$ | $1.1783$ | $0.950$ | $0.763$ | **$+0.1874$** | $100.0\%$ |
| **5** | $0.00043$ | $-6.096$ | $-6.091$ | $0.1989$ | $0.994$ | $0.728$ | **$+0.2659$** | $100.0\%$ |
| **10** | $\mathbf{0.00000}$ | $-6.392$ | $-6.392$ | $0.0108$ | $\mathbf{1.000}$ | $0.724$ | $\mathbf{+0.2762}$ | $\mathbf{100.0\%}$ |
| **25** | $\mathbf{0.00000}$ | $-6.400$ | $-6.400$ | $0.0000$ | $\mathbf{1.000}$ | $0.723$ | $\mathbf{+0.2766}$ | $\mathbf{100.0\%}$ |

#### Final Transition Breakdown
* $AB \to C$: $S_{\text{target}} = 1.000$, $S_{\text{competitor}} = 0.723 \implies M = \mathbf{+0.2766}$ `[PASSED M > 0]`
* $BC \to D$: $S_{\text{target}} = 1.000$, $S_{\text{competitor}} = 0.723 \implies M = \mathbf{+0.2766}$ `[PASSED M > 0]`
* $CD \to A$: $S_{\text{target}} = 1.000$, $S_{\text{competitor}} = 0.723 \implies M = \mathbf{+0.2766}$ `[PASSED M > 0]`
* $DA \to B$: $S_{\text{target}} = 1.000$, $S_{\text{competitor}} = 0.723 \implies M = \mathbf{+0.2766}$ `[PASSED M > 0]`

#### Generative Rollout from Seed `"AB"`
$$\mathbf{\text{"ABCDABCDAB"}}$$
*(Flawless deterministic multi-character autoregressive sequence).*

---

## 4. Key Architectural Insights & Fixes

1. **Why $W_{\text{rec}}$ Blew Up in Earlier Tests**:
   - In unconstrained recurrent reservoirs, updating $W_{\text{rec}} \leftarrow W_{\text{rec}} + \frac{\eta}{\beta}(h^\beta (h^\beta)^H - h^0 (h^0)^H)$ without spectral decay caused $\|W_{\text{rec}}\|$ to surge past $450$.
   - The reservoir entered autonomous limit-cycle oscillations that completely overpowered the feedforward input $W_{\text{in}} x$, collapsing all rollouts into the strongest self-reinforcing attractor.
2. **The Direct Physical Crossbar Theorem**:
   - For spatial wave sequences, the contrastive Equilibrium Propagation update on the crossbar conductances:
     $$\Delta W = \frac{\eta}{\beta} (y^\beta - y^0) x^H = \eta (y^* - y^0) x^H$$
     monotonically reshapes the energy landscape, creating a separate global minimum for each distinct input wave.
3. **The Baseline Competitor Ceiling ($0.723$)**:
   - In adjacent ASCII characters (such as A, B, C, D), the dual-harmonic acoustic formant encoding has an intrinsic carrier cross-talk of $|\langle \phi_A | \phi_B \rangle| = 0.7234$.
   - By driving target overlap to $S_{\text{target}} \to 1.000$, the Separation Margin reaches its maximum theoretical ceiling:
     $$M_{\max} = 1.000 - 0.7234 = \mathbf{+0.2766}$$
   - This proves that **learning reshapes the energy landscape in the correct direction**, definitively solving the attractor collapse on synthetic transitions.

---

## 5. Next Step in Research Roadmap

```
PASS-0: Numerical & Physical Invariants (Complete)
    ↓
PASS-1: Wave Transducer Validation (Complete)
    ↓
EP-01: Synthetic Transition Learning & Margin M > 0 Validation (COMPLETE)
    ├── Task 1 (1-gram cycle): M = +0.2761, Acc = 100.0% [PASSED]
    └── Task 2 (2-gram cycle): M = +0.2766, Acc = 100.0% [PASSED]
        ↓
EP-02: Dyck Grammar & Cavity Resonance Learning (NEXT)
    └── Stackless nested bracket validation with memristive gating
        ↓
EP-03: Semantic Associative Infilling Transitions
        ↓
EP-04: Natural Language Autoregressive Next-Wave Scaling
        ↓
Tier-4: Comparative Scaling vs Transformer / SSM
```
