import numpy as np
from src.transducer import GaborWaveTransducer
from benchmarks.ep01_learning_dynamics import PhysicalCrossbarLayer

def main():
    t = GaborWaveTransducer(n_grid=256, x_min=-10.0, x_max=10.0)
    corpus = "THE CAT IS SMALL. THE DOG IS FAST. THE SKY IS BLUE. "
    k = 4
    transitions = [(corpus[i:i+k], corpus[i+k]) for i in range(len(corpus)-k)]
    layer = PhysicalCrossbarLayer(256, 256, eta=0.08, beta=0.35)
    for epoch in range(120):
        for ctx, tgt in transitions:
            x = t.encode(ctx)
            tgt_w = t.encode(tgt)
            layer.train_step(x, tgt_w)
    
    seeds = ["THE C", "THE D", "THE S"]
    for seed in seeds:
        gen = seed
        for _ in range(20):
            ctx = gen[-k:]
            pred = layer.predict(t.encode(ctx))
            pred /= np.sqrt(np.sum(np.abs(pred)**2)*t.dx)
            char = t.decode(pred, expected_length=1)
            gen += char
        print(f"Seed '{seed}' -> '{gen}'")

if __name__ == "__main__":
    main()
