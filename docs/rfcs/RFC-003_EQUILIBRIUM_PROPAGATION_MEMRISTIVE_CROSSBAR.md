# RFC-003: Equilibrium Propagation on Memristive Crossbars
## Subsystem Specification: Local Physical Learning Without Backpropagation

* **Status:** `ACTIVE` (Implemented & Verified in [`src/equilibrium_propagation.py`](../../src/equilibrium_propagation.py))
* **Author:** Project Resonon / PhysLM
* **Base Document:** [Mathematical Formulation 02](../backbone/02_MATHEMATICAL_AND_PHYSICAL_FORMULATION.md)

---

## 1. Problem Statement

Standard Deep Learning requires **Reverse-Mode Automatic Differentiation (Backpropagation)**:
1. High memory footprint: All intermediate layer activations must be cached in VRAM for the reverse pass.
2. The *Weight Transport Problem*: Backpropagation requires separate, non-physical reverse circuits to transmit error gradients symmetrically backwards, which is physically impossible on solid-state analog hardware.

---

## 2. Specification: Two-Phase Equilibrium Propagation Protocol

RFC-003 specifies local energy-based adaptation on complex crossbars using **Equilibrium Propagation (Scellier & Bengio, 2017)**.

```mermaid
flowchart LR
    subgraph Free_Phase [Phase 1: Free Phase]
        In[Clamp Input x] --> Relax0[Settle to Free Equilibrium: s^0]
    end
    subgraph Nudged_Phase [Phase 2: Nudged Phase]
        Relax0 --> Nudge[Apply Output Perturbation: β * (y* - y)]
        Nudge --> RelaxB[Settle to Nudged Equilibrium: s^β]
    end
    subgraph Parameter_Update [Phase 3: Local Update]
        RelaxB --> Hebb["ΔW_ij = (η/β) * (s_i^β s_j^β* - s_i^0 s_j^0*)"]
    end
```

### 2.1 Phase-Preserving Complex Saturation
To prevent energy explosion while preserving quantum phase angles $\arg(z)$ in Hilbert space:

$$f(z) = \frac{z}{1.0 + |z|}, \quad |f(z)| < 1.0, \quad \arg(f(z)) \equiv \arg(z)$$

### 2.2 Free and Nudged Phase Dynamics
* **Free Phase:** The input $\mathbf{x}$ is clamped. Hidden nodes $\mathbf{h}$ and output nodes $\mathbf{y}$ relax:
  $$\frac{d\mathbf{h}}{d\tau} = f(W_{\text{in}} \mathbf{x} + W_{\text{rec}} \mathbf{h} + W_{\text{out}}^\dagger \mathbf{y}) - \mathbf{h}$$
  $$\frac{d\mathbf{y}}{d\tau} = f(W_{\text{out}} \mathbf{h}) - \mathbf{y}$$
  Settling to $(\mathbf{h}^0, \mathbf{y}^0)$.

* **Nudged Phase:** A weak target perturbation $\beta (\mathbf{y}^* - \mathbf{y})$ is injected at the output:
  $$\frac{d\mathbf{y}}{d\tau} = f(W_{\text{out}} \mathbf{h}) - \mathbf{y} + \beta (\mathbf{y}^* - \mathbf{y})$$
  Settling to $(\mathbf{h}^\beta, \mathbf{y}^\beta)$.

### 2.3 Contrastive Hebbian Crossbar Update
Parameter updates occur **purely locally at the wire junctions** of the physical crossbar:

$$\Delta W_{\text{out}} = \frac{\eta}{\beta} \left( \mathbf{y}^\beta (\mathbf{h}^\beta)^\dagger - \mathbf{y}^0 (\mathbf{h}^0)^\dagger \right)$$
$$\Delta W_{\text{in}} = \frac{\eta}{\beta} \left( \mathbf{h}^\beta \mathbf{x}^\dagger - \mathbf{h}^0 \mathbf{x}^\dagger \right)$$
$$\Delta W_{\text{rec}} = \frac{\eta}{\beta} \left( \mathbf{h}^\beta (\mathbf{h}^\beta)^\dagger - \mathbf{h}^0 (\mathbf{h}^0)^\dagger \right)$$

No global backpropagation or activation tensor graphs are ever stored.
