"""Run a short GA per preset and save the top-1 grid for FIG B.1.

Writes:
    <out_root>/<preset_name>/grid.npy
    <out_root>/<preset_name>/preset.json   {"name": <display name>}

Small GA config (pop=80, gens=20) is enough for the gallery — the B.1
figure is illustrative, not a performance claim.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np


PRESET_FNS = [
    "bandpass_5ghz", "lowpass_10ghz", "broadband_match", "notch_10ghz",
    "directional_coupler_10ghz", "hybrid_coupler_10ghz",
    "power_divider_10ghz", "crossover_10ghz", "diplexer_5_15ghz",
    "antenna_match_28ghz",
    "narrowband_match_10ghz", "wideband_match_5_15ghz",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt",
        default=os.environ.get("INVDESIGN_CKPT", "./checkpoints/best_model.pt"))
    ap.add_argument("--out-root", default="./preset_gallery")
    ap.add_argument("--pop-size", type=int, default=80)
    ap.add_argument("--n-generations", type=int, default=20)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from optimizer.ga import GAConfig, GeneticAlgorithm, SurrogateEvaluator
    from optimizer import objectives as obj_mod
    import torch, gc

    ev = SurrogateEvaluator(args.ckpt)
    root = Path(args.out_root); root.mkdir(parents=True, exist_ok=True)

    for fn_name in PRESET_FNS:
        fn = getattr(obj_mod, fn_name, None)
        if fn is None:
            print(f"[skip] missing preset function {fn_name}")
            continue
        objective = fn()
        display = getattr(objective, "description",
                          getattr(objective, "name", fn_name))
        sub = root / fn_name; sub.mkdir(parents=True, exist_ok=True)
        if (sub / "grid.npy").exists():
            print(f"[skip] already built: {fn_name}")
            continue

        cfg = GAConfig(pop_size=args.pop_size,
                       n_generations=args.n_generations,
                       seed=args.seed, local_search=True, local_search_k=3)
        ga = GeneticAlgorithm(objective, ev, cfg)
        res = ga.run(verbose=False)
        grid = res["top_k"][0]["grid"]
        fit  = float(res["best_fitness_history"][-1])
        np.save(sub / "grid.npy", grid.astype(np.int8))
        (sub / "preset.json").write_text(json.dumps({
            "name": display, "fn": fn_name,
            "best_fitness": fit,
            "fill": float(grid[1:-1, 1:-1].mean()),
        }, indent=2))
        print(f"[ok] {fn_name:30s}  fit={fit:+.1f}  fill={grid[1:-1,1:-1].mean():.0%}")


if __name__ == "__main__":
    main()
