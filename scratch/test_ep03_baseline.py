from src.transducer import GaborWaveTransducer
from src.associative_memory import ContinuousAssociativeMemory
import numpy as np

t = GaborWaveTransducer(n_grid=512, x_min=-15.0, x_max=15.0)
mem = ContinuousAssociativeMemory(transducer=t, beta=15.0, alpha_clamp=0.8, dt=0.04)

indonesian_pairs = [
    ("kucing:meong", "kucing:", 12),
    ("langit:biru", "langit:", 11),
    ("api:panas", "api:", 9),
    ("es:dingin", "es:", 9),
    ("matahari:terang", "matahari:", 15)
]

for full, _, _ in indonesian_pairs:
    mem.store(full)

print(f"Stored {len(mem.patterns)} Indonesian concept pairs.")

print("\n--- TEST 1: Canonical Clean Infilling ---")
for full, prompt, total_len in indonesian_pairs:
    completion, energy = mem.complete(prompt, total_expected_length=total_len, steps=120)
    psi_relaxed = t.encode(completion)
    target_idx = mem.pattern_texts.index(full)
    overlaps = [mem.compute_overlap(p, psi_relaxed) for p in mem.patterns]
    s_tgt = overlaps[target_idx]
    overlaps[target_idx] = -1.0
    s_comp = max(overlaps)
    margin = s_tgt - s_comp
    print(f"Prompt '{prompt}' -> '{completion}' | E={energy:.4f} | Margin M={margin:+.4f} {'[OK]' if completion == full else '[FAIL]'}")

print("\n--- TEST 2: Generalization Under Input Thermal Noise ---")
for noise in [0.05, 0.10, 0.15, 0.20]:
    prompt = "kucing:"
    expected = "kucing:meong"
    completion, energy = mem.complete(prompt, total_expected_length=12, steps=120, noise_sigma=noise)
    psi_relaxed = t.encode(completion)
    target_idx = mem.pattern_texts.index(expected)
    overlaps = [mem.compute_overlap(p, psi_relaxed) for p in mem.patterns]
    s_tgt = overlaps[target_idx]
    overlaps[target_idx] = -1.0
    s_comp = max(overlaps)
    margin = s_tgt - s_comp
    print(f"Noise sigma={noise:.2f} | Result: '{completion}' | Margin M={margin:+.4f} {'[OK]' if completion == expected else '[FAIL]'}")

print("\n--- TEST 3: Generalization to Prefix Variations (Shorter Context) ---")
variations = [
    ("kucing :", "kucing:meong", 12),
    ("kucing", "kucing:meong", 12),
    ("kucin", "kucing:meong", 12),
    ("kuc", "kucing:meong", 12)
]
for p_var, expected, total_len in variations:
    completion, energy = mem.complete(p_var, total_expected_length=total_len, steps=120)
    psi_relaxed = t.encode(completion)
    target_idx = mem.pattern_texts.index(expected)
    overlaps = [mem.compute_overlap(p, psi_relaxed) for p in mem.patterns]
    s_tgt = overlaps[target_idx]
    overlaps[target_idx] = -1.0
    s_comp = max(overlaps)
    margin = s_tgt - s_comp
    print(f"Variant '{p_var:<8}' -> '{completion}' | Margin M={margin:+.4f} {'[OK]' if completion == expected else '[FAIL]'}")

print("\n--- TEST 4: OOD Distractor Rejection (Unseen Concepts) ---")
distractors = ["batu:", "air:", "pohon:"]
for dist in distractors:
    completion, energy = mem.complete(dist, total_expected_length=10, steps=120)
    psi_relaxed = t.encode(completion)
    overlaps = [mem.compute_overlap(p, psi_relaxed) for p in mem.patterns]
    max_ov = max(overlaps)
    print(f"Distractor '{dist}' -> Relaxed to: '{completion}' | Max Overlap={max_ov:.3f} | Energy={energy:.4f} (High energy indicates rejection)")
