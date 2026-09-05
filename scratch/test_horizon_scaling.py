import numpy as np
from src.transducer import GaborWaveTransducer
from benchmarks.ep01_learning_dynamics import PhysicalCrossbarLayer

def main():
    t = GaborWaveTransducer(n_grid=256, x_min=-10.0, x_max=10.0)
    corpus = "THE CAT IS SMALL. THE DOG IS FAST. THE SKY IS BLUE. "
    k = 4
    transitions = [(corpus[i:i+k], corpus[i+k]) for i in range(len(corpus)-k)]
    layer = PhysicalCrossbarLayer(256, 256, eta=0.08, beta=0.35)
    
    # Train crossbar
    for epoch in range(120):
        for ctx, tgt in transitions:
            x = t.encode(ctx)
            tgt_w = t.encode(tgt)
            layer.train_step(x, tgt_w)
            
    # Reference sequence of length 260
    ref_full = (corpus * 10)[:265]
    seed = ref_full[:k]
    
    horizons = [1, 4, 16, 64, 256]
    
    for mode in ["projective", "free_flight"]:
        print(f"\n================ Mode: {mode.upper()} ================")
        print(f"{'H':<6}{'L(H)':<12}{'E(H)':<12}{'R_phi(H)':<12}{'Delta_drift':<14}{'Delta_basis':<14}{'VCR(%)':<10}")
        print("-" * 75)
        
        # Initialize slots for free flight
        wave_slots = [t.encode(c) for c in seed]
        total_span = (k - 1) * t.char_spacing
        x_start = -total_span / 2.0
        x_positions = [x_start + j * t.char_spacing for j in range(k)]
        
        current_text = seed
        sq_errors = []
        vcr_count = 0
        
        for step in range(256):
            true_char = ref_full[k + step]
            true_wave = t.encode(true_char)
            
            if mode == "projective":
                ctx = current_text[-k:]
                pred = layer.predict(t.encode(ctx))
                pred_norm = np.sqrt(np.sum(np.abs(pred)**2) * t.dx)
                if pred_norm > 1e-12:
                    pred /= pred_norm
                char = t.decode(pred, expected_length=1)
                current_text += char
            else:
                # Free flight
                ctx_wave = np.zeros(t.n_grid, dtype=complex)
                for j in range(k):
                    env = np.exp(-((t.x - x_positions[j])**2) / (2.0 * (t.sigma**2)))
                    ctx_wave += env * wave_slots[j]
                ctx_norm = np.sqrt(np.sum(np.abs(ctx_wave)**2) * t.dx)
                if ctx_norm > 1e-12:
                    ctx_wave /= ctx_norm
                pred = layer.predict(ctx_wave)
                pred_norm = np.sqrt(np.sum(np.abs(pred)**2) * t.dx)
                if pred_norm > 1e-12:
                    pred /= pred_norm
                char = t.decode(pred, expected_length=1)
                current_text += char
                wave_slots = wave_slots[1:] + [pred]
                
            # Error and telemetry
            err_sq = float(np.sum(np.abs(pred - true_wave)**2) * t.dx)
            sq_errors.append(err_sq)
            
            # Basis overlap & VCR
            overlaps = [np.abs(np.sum(np.conj(t.basis_probe(0.0, c)) * pred) * t.dx) for c in t.charset]
            max_ov = max(overlaps)
            if char in t.charset and max_ov > 0.40:
                vcr_count += 1
                
            current_H = step + 1
            if current_H in horizons:
                l_h = float(np.mean(sq_errors))
                e_h = err_sq
                r_phi = float(np.abs(np.sum(np.conj(true_wave) * pred) * t.dx))
                d_drift = 1.0 - r_phi
                d_basis = 1.0 - max_ov
                vcr = (vcr_count / current_H) * 100.0
                print(f"{current_H:<6}{l_h:<12.4f}{e_h:<12.4f}{r_phi:<12.4f}{d_drift:<14.4f}{d_basis:<14.4f}{vcr:<10.1f}")
                
        print(f"Generated text sample (first 50 chars): {repr(current_text[:50])}")

if __name__ == "__main__":
    main()
