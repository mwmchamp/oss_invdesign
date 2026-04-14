"""Main optimizer: GA search + optional OpenEMS validation.

Usage:
    # Quick test (surrogate only)
    python -m optimizer.run --objective bandpass_5ghz --generations 100 --no-validate

    # Full run with validation
    python -m optimizer.run --objective broadband_match --generations 300

    # List available objectives
    python -m optimizer.run --list-objectives
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from optimizer.ga import GeneticAlgorithm, GAConfig, SurrogateEvaluator
from optimizer.objectives import OBJECTIVES, MatchingObjective
from optimizer.validate import validate_top_k


def save_results(results: dict, output_dir: Path) -> None:
    """Save GA results: grids, S-params, fitness history, plots."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save top-K grids
    for r in results["top_k"]:
        rank = r["rank"]
        np.save(output_dir / f"grid_rank{rank:02d}.npy", r["grid"])
        np.save(output_dir / f"sparams_rank{rank:02d}.npy", r["sparams"])

    # Save fitness history
    history = {
        "objective": results["objective"],
        "n_generations": results["n_generations"],
        "time_s": results["time_s"],
        "best_fitness_history": results["best_fitness_history"],
        "mean_fitness_history": results["mean_fitness_history"],
        "top_k_summary": [
            {
                "rank": r["rank"],
                "fitness": r["fitness"],
                "fill": r["fill"],
                "goals": r["goals"],
            }
            for r in results["top_k"]
        ],
    }
    with (output_dir / "ga_results.json").open("w") as f:
        json.dump(history, f, indent=2)

    # Plot convergence
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Fitness convergence
        ax = axes[0]
        gens = range(len(results["best_fitness_history"]))
        ax.plot(gens, results["best_fitness_history"], "b-", label="Best")
        ax.plot(gens, results["mean_fitness_history"], "r-", alpha=0.5, label="Mean")
        ax.set_xlabel("Generation")
        ax.set_ylabel("Fitness (higher=better)")
        ax.set_title(f"GA Convergence — {results['objective']}")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Best design S-params
        ax = axes[1]
        best = results["top_k"][0]
        sparams = best["sparams"]
        freq_ghz = np.linspace(1, 30, sparams.shape[0])

        for i, j, label, ls in [
            (0, 0, "|S11|", "-"),
            (0, 1, "|S21|", "--"),
            (0, 2, "|S31|", ":"),
            (0, 3, "|S41|", "-."),
        ]:
            mag_db = 20 * np.log10(np.abs(sparams[:, i, j]) + 1e-12)
            ax.plot(freq_ghz, mag_db, ls, label=label, linewidth=1.5)

        ax.set_xlabel("Frequency (GHz)")
        ax.set_ylabel("|S| (dB)")
        ax.set_title(f"Best Design — fitness={best['fitness']:.4f}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-60, 5)

        plt.tight_layout()
        fig.savefig(output_dir / "ga_results.png", dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved plots to {output_dir}/ga_results.png")
    except ImportError:
        print("matplotlib not available, skipping plots")


def main():
    parser = argparse.ArgumentParser(description="Pixel grid inverse design optimizer")
    parser.add_argument("--objective", type=str, default="broadband_match",
                        choices=list(OBJECTIVES.keys()),
                        help="Matching network objective")
    parser.add_argument("--list-objectives", action="store_true",
                        help="List available objectives and exit")
    parser.add_argument("--checkpoint", type=str,
                        default=os.environ.get(
                            "INVDESIGN_CKPT",
                            "./checkpoints/best_model.pt"),
                        help="Path to trained surrogate checkpoint")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: optimizer/results/<objective>)")
    parser.add_argument("--pop-size", type=int, default=200)
    parser.add_argument("--generations", type=int, default=300)
    parser.add_argument("--mutation-rate", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--no-validate", action="store_true",
                        help="Skip OpenEMS validation of top candidates")
    parser.add_argument("--validate-k", type=int, default=3,
                        help="Number of top candidates to validate with OpenEMS")
    parser.add_argument("--low-fidelity-validate", action="store_true",
                        help="Use standard (not high-fidelity) OpenEMS for validation")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    if args.list_objectives:
        from optimizer.objectives import ImpedanceMatchObjective, ImpedanceMatchGoal
        print("Available objectives:")
        for name, factory in OBJECTIVES.items():
            obj = factory()
            print(f"  {name}: {obj.description}")
            if isinstance(obj, ImpedanceMatchObjective):
                for g in obj.goals:
                    print(f"    match Zs={g.z_source} → Zl={g.z_load} "
                          f"@ {g.f_min_ghz:.0f}-{g.f_max_ghz:.0f}GHz (w={g.weight})")
                for g in obj.sparam_goals:
                    print(f"    |S{g.i+1}{g.j+1}| {g.mode} {g.target_db:.0f}dB "
                          f"@ {g.f_min_ghz:.0f}-{g.f_max_ghz:.0f}GHz (w={g.weight})")
            else:
                for g in obj.goals:
                    print(f"    |S{g.i+1}{g.j+1}| {g.mode} {g.target_db:.0f}dB "
                          f"@ {g.f_min_ghz:.0f}-{g.f_max_ghz:.0f}GHz (w={g.weight})")
        return

    # Setup
    objective = OBJECTIVES[args.objective]()
    output_dir = Path(args.output_dir or f"optimizer/results/{args.objective}")

    print(f"=== Pixel Grid Inverse Design ===")
    print(f"Objective: {objective.name} — {objective.description}")
    print(f"Population: {args.pop_size}, Generations: {args.generations}")
    print(f"Output: {output_dir}")

    # Load surrogate
    print(f"\nLoading surrogate from {args.checkpoint}...")
    evaluator = SurrogateEvaluator(args.checkpoint, device=args.device)
    print("  Surrogate loaded.")

    # Configure GA
    config = GAConfig(
        pop_size=args.pop_size,
        n_generations=args.generations,
        mutation_rate=args.mutation_rate,
        seed=args.seed,
    )

    # Run GA
    print(f"\nRunning GA optimization...")
    ga = GeneticAlgorithm(objective, evaluator, config)
    results = ga.run(verbose=True)

    # Save results
    save_results(results, output_dir)
    print(f"\nResults saved to {output_dir}/")

    # OpenEMS validation
    if not args.no_validate:
        print(f"\n=== OpenEMS Validation (top {args.validate_k} candidates) ===")
        val_dir = output_dir / "validation"
        validated = validate_top_k(
            results, objective, val_dir,
            k=args.validate_k,
            high_fidelity=not args.low_fidelity_validate,
        )

        # Compare surrogate vs OpenEMS
        print("\n=== Surrogate vs OpenEMS Comparison ===")
        print(f"{'Rank':>4} | {'Surrogate':>10} | {'OpenEMS':>10} | {'Delta':>8}")
        print("-" * 40)
        for v in validated:
            if v["success"]:
                sf = v["surrogate_fitness"]
                of = v["objective_score"]["fitness"]
                print(f"{v['rank']:>4} | {sf:>10.4f} | {of:>10.4f} | {of-sf:>8.4f}")
    else:
        print("\nSkipping OpenEMS validation (use without --no-validate to enable)")


if __name__ == "__main__":
    main()
