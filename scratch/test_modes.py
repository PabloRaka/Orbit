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
            
    # Test Mode B: Projective Restoration
    print("--- Mode B: Projective Restoration ---")
    gen_b = "THE "
    for _ in range(16):
        ctx = gen_b[-k:]
        pred = layer.predict(t.encode(ctx))
        pred /= np.sqrt(np.sum(np.abs(pred)**2)*t.dx)
        char = t.decode(pred, expected_length=1)
        gen_b += char
    print("Mode B Rollout:", repr(gen_b))

    # Test Mode A: Free Flight
    print("--- Mode A: Continuous Analog Free-Flight ---")
    # In Mode A, we maintain a list of k wave states
    # Initialize from seed "THE "
    wave_slots = [t.encode(c) for c in "THE "]
    
    # Precompute slot center positions for k=4
    total_span = (k - 1) * t.char_spacing
    x_start = -total_span / 2.0
    x_positions = [x_start + j * t.char_spacing for j in range(k)]
    
    gen_a = "THE "
    for step in range(16):
        # Assemble context wave by placing each slot at its spatial position
        ctx_wave = np.zeros(t.n_grid, dtype=complex)
        for j in range(k):
            env = np.exp(-((t.x - x_positions[j])**2) / (2.0 * (t.sigma**2)))
            ctx_wave += env * wave_slots[j]
        ctx_norm = np.sqrt(np.sum(np.abs(ctx_wave)**2) * t.dx)
        if ctx_norm > 1e-12:
            ctx_wave /= ctx_norm
            
        # Predict next wave
        pred = layer.predict(ctx_wave)
        pred_norm = np.sqrt(np.sum(np.abs(pred)**2) * t.dx)
        if pred_norm > 1e-12:
            pred /= pred_norm
            
        # Telemetry only: observe what character this is closest to
        char_obs = t.decode(pred, expected_length=1)
        gen_a += char_obs
        
        # Free-flight update: push uncollapsed pred into wave_slots without discrete collapse!
        wave_slots = wave_slots[1:] + [pred]
        
    print("Mode A Rollout:", repr(gen_a))

    print("\n--- Held-Out Prompt Generalization ---")
    for prompt in ["THE CAT IS F", "THE DOG IS S", "THE DOG IS B"]:
        gen = prompt
        for _ in range(10):
            ctx = gen[-k:]
            pred = layer.predict(t.encode(ctx))
            pred /= np.sqrt(np.sum(np.abs(pred)**2)*t.dx)
            char = t.decode(pred, expected_length=1)
            gen += char
            if char == '.':
                break
        print(f"Held-out prompt '{prompt}' -> '{gen}'")

if __name__ == "__main__":
    main()

