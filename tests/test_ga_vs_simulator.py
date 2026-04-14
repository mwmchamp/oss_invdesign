"""Test GA-optimized designs against FDTD simulator.

Generates random test cases, runs the GA optimizer with the trained surrogate,
then validates the top designs against OpenEMS FDTD. Reports the surrogate-FDTD
gap for each design.

Usage:
    python tests/test_ga_vs_simulator.py \
        --checkpoint /path/to/best_model.pt \
        --objective bandpass_5ghz \
        --output-dir /path/to/test_output \
        --n-tests 5

For SLURM submission, use tests/run_ga_test.sh.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from optimizer.ga import GAConfig, GeneticAlgorithm, SurrogateEvaluator, EnsembleEvaluator
from optimizer.objectives import OBJECTIVES
from invdesign.simulate import SimConfig, simulate_grid


def run_single_test(
    objective_name: str,
    evaluator: SurrogateEvaluator,
    output_dir: Path,
    ga_seed: int = 1,
    n_ga_generations: int = 100,
    pop_size: int = 100,
    n_validate: int = 3,
    verbose: bool = True,
) -> dict:
    """Run one GA optimization + FDTD validation test.

    Args:
        objective_name: key from OBJECTIVES registry
        evaluator: trained surrogate model
        output_dir: where to save results
        ga_seed: random seed for GA
        n_ga_generations: number of GA generations (reduced for testing)
        pop_size: GA population size (reduced for testing)
        n_validate: number of top designs to validate with FDTD
        verbose: print progress

    Returns:
        dict with surrogate vs FDTD comparison results
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    objective = OBJECTIVES[objective_name]()

    config = GAConfig(
        pop_size=pop_size,
        n_generations=n_ga_generations,
        seed=ga_seed,
        local_search=True,
        local_search_k=min(3, n_validate),
    )

    if verbose:
        print(f"\n{'='*60}")
        print(f"Test: {objective_name} (seed={ga_seed})")
        print(f"{'='*60}")

    # Run GA
    t_ga = time.time()
    ga = GeneticAlgorithm(objective, evaluator, config)
    ga_results = ga.run(verbose=verbose)
    dt_ga = time.time() - t_ga

    if verbose:
        print(f"\nGA completed in {dt_ga:.1f}s")
        print(f"Best surrogate fitness: {ga_results['top_k'][0]['fitness']:.4f}")

    # Validate top designs with FDTD
    sim_config = SimConfig(
        nr_ts=500_000,
        end_criteria_db=-40.0,
        n_freq=30,
        f_start=1e9,
        f_stop=30e9,
        mesh_div=20.0,
        use_pml=True,
    )

    validation_results = []
    for i, design in enumerate(ga_results["top_k"][:n_validate]):
        grid = design["grid"]
        surr_fitness = design["fitness"]
        surr_sparams = design["sparams"]

        if verbose:
            print(f"\n  Validating design {i+1}/{n_validate} "
                  f"(surrogate fitness: {surr_fitness:.4f})...")

        # Save grid
        np.save(output_dir / f"grid_{i:02d}.npy", grid)

        # Run FDTD
        work_dir = output_dir / f"fdtd_{i:02d}"
        t_fdtd = time.time()
        try:
            result = simulate_grid(grid, work_dir, sim_config)
            dt_fdtd = time.time() - t_fdtd

            if not result["success"]:
                if verbose:
                    print(f"    FDTD failed: {result.get('error', 'unknown')}")
                validation_results.append({
                    "rank": i,
                    "surrogate_fitness": surr_fitness,
                    "fdtd_fitness": None,
                    "fdtd_error": result.get("error", "unknown"),
                    "fdtd_time_s": dt_fdtd,
                })
                continue

            fdtd_sparams = result["sparams"]
            fdtd_score = objective.evaluate(fdtd_sparams)
            fdtd_fitness = fdtd_score["fitness"]

            # Compute S-param comparison metrics
            surr_mag = np.abs(surr_sparams)
            fdtd_mag = np.abs(fdtd_sparams)
            eps = 1e-12
            surr_db = 20 * np.log10(surr_mag + eps)
            fdtd_db = 20 * np.log10(fdtd_mag + eps)
            db_error = np.abs(surr_db - fdtd_db)

            gap_ratio = fdtd_fitness / surr_fitness if surr_fitness != 0 else float("inf")

            vr = {
                "rank": i,
                "surrogate_fitness": float(surr_fitness),
                "fdtd_fitness": float(fdtd_fitness),
                "gap_ratio": float(gap_ratio),
                "mean_db_error": float(db_error.mean()),
                "max_db_error": float(db_error.max()),
                "p90_db_error": float(np.percentile(db_error, 90)),
                "fdtd_time_s": dt_fdtd,
                "fill": float(grid[1:-1, 1:-1].mean()),
            }
            validation_results.append(vr)

            # Save FDTD sparams
            np.save(output_dir / f"sparams_fdtd_{i:02d}.npy", fdtd_sparams)
            np.save(output_dir / f"sparams_surr_{i:02d}.npy", surr_sparams)

            if verbose:
                print(f"    FDTD fitness: {fdtd_fitness:.4f} "
                      f"(gap: {gap_ratio:.2f}x, {dt_fdtd:.0f}s)")
                print(f"    Mean |dB| error: {db_error.mean():.2f} dB")
                for g in fdtd_score["goals"]:
                    print(f"      {g['goal']}: {g['achieved_db']:.1f}dB")

        except Exception as e:
            dt_fdtd = time.time() - t_fdtd
            if verbose:
                print(f"    FDTD exception: {e}")
            validation_results.append({
                "rank": i,
                "surrogate_fitness": surr_fitness,
                "fdtd_fitness": None,
                "fdtd_error": str(e),
                "fdtd_time_s": dt_fdtd,
            })

    # Summary
    test_result = {
        "objective": objective_name,
        "ga_seed": ga_seed,
        "ga_time_s": dt_ga,
        "n_generations": n_ga_generations,
        "pop_size": pop_size,
        "best_surrogate_fitness": float(ga_results["top_k"][0]["fitness"]),
        "validations": validation_results,
    }

    # Save results
    with (output_dir / "results.json").open("w") as f:
        json.dump(test_result, f, indent=2, default=str)

    if verbose:
        valid = [v for v in validation_results if v.get("fdtd_fitness") is not None]
        if valid:
            gaps = [v["gap_ratio"] for v in valid]
            db_errs = [v["mean_db_error"] for v in valid]
            print(f"\n  Summary for {objective_name}:")
            print(f"    Validated: {len(valid)}/{n_validate}")
            print(f"    Gap ratios: {[f'{g:.2f}x' for g in gaps]}")
            print(f"    Mean dB errors: {[f'{e:.1f}' for e in db_errs]}")
            best_fdtd = max(valid, key=lambda v: v["fdtd_fitness"])
            print(f"    Best FDTD fitness: {best_fdtd['fdtd_fitness']:.4f} "
                  f"(surr: {best_fdtd['surrogate_fitness']:.4f})")

    return test_result


def main():
    parser = argparse.ArgumentParser(description="Test GA vs FDTD simulator")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to surrogate model checkpoint (.pt)")
    parser.add_argument("--ensemble-dir", type=str, default=None,
                        help="Directory with ensemble member checkpoints (member_*/best_model.pt)")
    parser.add_argument("--objective", type=str, default="bandpass_5ghz",
                        choices=list(OBJECTIVES.keys()),
                        help="Objective to test (default: bandpass_5ghz)")
    parser.add_argument("--objectives", type=str, nargs="+", default=None,
                        help="Multiple objectives to test")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--n-tests", type=int, default=3,
                        help="Number of random seed tests per objective")
    parser.add_argument("--n-validate", type=int, default=3,
                        help="Top-K designs to validate with FDTD per test")
    parser.add_argument("--ga-generations", type=int, default=100)
    parser.add_argument("--pop-size", type=int, default=100)
    parser.add_argument("--base-seed", type=int, default=1)
    args = parser.parse_args()

    # Load evaluator
    if args.ensemble_dir:
        import glob
        ckpts = sorted(glob.glob(f"{args.ensemble_dir}/member_*/best_model.pt"))
        if not ckpts:
            raise FileNotFoundError(f"No ensemble checkpoints in {args.ensemble_dir}")
        print(f"Loading ensemble with {len(ckpts)} members")
        evaluator = EnsembleEvaluator(ckpts)
    else:
        print(f"Loading single model: {args.checkpoint}")
        evaluator = SurrogateEvaluator(args.checkpoint)

    objectives = args.objectives or [args.objective]
    output_base = Path(args.output_dir)
    all_results = []

    for obj_name in objectives:
        for test_idx in range(args.n_tests):
            seed = args.base_seed + test_idx * 1000
            test_dir = output_base / f"{obj_name}_seed{seed}"

            result = run_single_test(
                objective_name=obj_name,
                evaluator=evaluator,
                output_dir=test_dir,
                ga_seed=seed,
                n_ga_generations=args.ga_generations,
                pop_size=args.pop_size,
                n_validate=args.n_validate,
            )
            all_results.append(result)

    # Overall summary
    print(f"\n{'='*60}")
    print("OVERALL SUMMARY")
    print(f"{'='*60}")
    for r in all_results:
        valid = [v for v in r["validations"] if v.get("fdtd_fitness") is not None]
        if valid:
            best = max(valid, key=lambda v: v["fdtd_fitness"])
            gaps = [v["gap_ratio"] for v in valid]
            print(f"  {r['objective']} (seed={r['ga_seed']}): "
                  f"surr={r['best_surrogate_fitness']:.4f} "
                  f"fdtd={best['fdtd_fitness']:.4f} "
                  f"gap={np.mean(gaps):.2f}x "
                  f"dB_err={np.mean([v['mean_db_error'] for v in valid]):.1f}")

    # Save aggregate
    with (output_base / "all_results.json").open("w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nAll results saved to {output_base / 'all_results.json'}")


if __name__ == "__main__":
    main()
