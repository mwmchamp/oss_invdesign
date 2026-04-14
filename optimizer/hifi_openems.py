"""High-fidelity OpenEMS validation for optimized pixel grid designs.

This module is NOT part of the optimization loop. It provides a final,
high-accuracy FDTD simulation for internal validation of the best designs
produced by the GA optimizer + surrogate pipeline.

Key differences from the dataset-generation simulation:
  - Finer mesh: mesh_div=40 (vs 20)
  - More timesteps: 5M (vs 500K)
  - Stricter end criteria: -60 dB (vs -35 dB)
  - More frequency points: 200 (vs 30)
  - Wider frequency range: 0.1-40 GHz (vs 1-30 GHz)

Usage:
    python -m optimizer.hifi_openems --grid path/to/grid.npy --objective broadband_match
    python -m optimizer.hifi_openems --results path/to/ga_results.npz --top-k 3
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from invdesign.simulate import SimConfig, simulate_grid, HIFI_CONFIG
from invdesign.openems_pixel import validate_sparams
from optimizer.objectives import MatchingObjective, OBJECTIVES


# Re-export for backward compat
HiFiConfig = SimConfig


def run_hifi_simulation(
    grid: np.ndarray,
    output_dir: Path,
    *,
    config: SimConfig | None = None,
    objective: MatchingObjective | None = None,
) -> dict:
    """Run a high-fidelity FDTD simulation on a single design."""
    config = config or HIFI_CONFIG
    output_dir = Path(output_dir).resolve()

    print(f"  Mesh div: {config.mesh_div}")
    print(f"  Timesteps: {config.nr_ts:,}")
    print(f"  End criteria: {config.end_criteria_db} dB")
    print(f"  Frequencies: {config.n_freq} ({config.f_start/1e9:.1f}-{config.f_stop/1e9:.1f} GHz)")
    print(f"  Boundaries: {'PML' if config.use_pml else 'PEC'}")

    result = simulate_grid(grid, output_dir, config)

    if result["success"]:
        result["validation_checks"] = validate_sparams(
            result["sparams"], passivity_tol=0.005, reciprocity_tol=0.02,
        )
        if objective is not None:
            hifi_freq_ghz = np.linspace(
                config.f_start / 1e9, config.f_stop / 1e9, config.n_freq,
            )
            result["objective_score"] = objective.evaluate(
                result["sparams"], freq_ghz=hifi_freq_ghz,
            )

    return result


def evaluate_ga_results(
    results_path: Path,
    objective_name: str,
    output_dir: Path,
    *,
    top_k: int = 3,
    config: SimConfig | None = None,
) -> list[dict]:
    """Run high-fidelity validation on the top-K designs from GA results."""
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    config = config or HIFI_CONFIG

    data = np.load(results_path, allow_pickle=True)
    top_k_results = data["top_k"].item() if "top_k" in data else data["top_k"]

    objective = OBJECTIVES[objective_name]()
    validated = []

    for i in range(min(top_k, len(top_k_results))):
        candidate = top_k_results[i]
        grid = candidate["grid"]
        surrogate_fitness = candidate["fitness"]

        print(f"\n{'='*60}")
        print(f"High-fidelity validation: candidate {i+1}/{top_k}")
        print(f"  Surrogate fitness: {surrogate_fitness:.4f}")
        print(f"  Fill factor: {candidate['fill']:.2%}")
        print(f"{'='*60}")

        work_dir = output_dir / f"candidate_{i:02d}"
        result = run_hifi_simulation(
            grid=grid, output_dir=work_dir, config=config, objective=objective,
        )
        result["rank"] = i
        result["surrogate_fitness"] = surrogate_fitness

        if result["success"]:
            hifi_fitness = result["objective_score"]["fitness"]
            print(f"\n  Results:")
            print(f"    Surrogate fitness: {surrogate_fitness:.4f}")
            print(f"    Hi-fi fitness:     {hifi_fitness:.4f}")
            print(f"    Gap:               {hifi_fitness - surrogate_fitness:.4f}")
            print(f"    Sim time:          {result['time_s']:.0f}s")
            print(f"    Passive:           {result['validation_checks']['is_passive']}")
            print(f"    Reciprocal:        {result['validation_checks']['is_reciprocal']}")
            for g in result["objective_score"]["goals"]:
                print(f"    {g['goal']}: {g['achieved_db']:.1f}dB "
                      f"(worst: {g.get('worst_db', float('nan')):.1f}dB)")
        else:
            print(f"\n  FAILED: {result.get('error', 'unknown')}")

        validated.append(result)

    # Save summary
    summary = []
    for v in validated:
        entry = {
            "rank": v["rank"],
            "success": v["success"],
            "surrogate_fitness": v["surrogate_fitness"],
            "time_s": v["time_s"],
        }
        if v["success"]:
            entry["hifi_fitness"] = v["objective_score"]["fitness"]
            entry["goals"] = v["objective_score"]["goals"]
            entry["validation_checks"] = {
                k: val for k, val in v["validation_checks"].items()
                if isinstance(val, (bool, int, float))
            }
        summary.append(entry)

    with (output_dir / "hifi_summary.json").open("w") as f:
        json.dump(summary, f, indent=2, default=str)

    return validated


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="High-fidelity OpenEMS validation for optimized designs"
    )
    parser.add_argument("--grid", type=str, help="Path to a single grid .npy file")
    parser.add_argument("--results", type=str, help="Path to GA results .npz file")
    parser.add_argument("--objective", type=str, required=True,
                        choices=list(OBJECTIVES.keys()))
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--nr-ts", type=int, default=5_000_000)
    parser.add_argument("--end-criteria-db", type=float, default=-60.0)
    parser.add_argument("--n-freq", type=int, default=200)
    parser.add_argument("--mesh-div", type=float, default=40.0)
    parser.add_argument("--no-pml", action="store_true")
    args = parser.parse_args()

    config = SimConfig(
        nr_ts=args.nr_ts,
        end_criteria_db=args.end_criteria_db,
        n_freq=args.n_freq,
        mesh_div=args.mesh_div,
        use_pml=not args.no_pml,
    )

    if args.grid:
        grid = np.load(args.grid)
        objective = OBJECTIVES[args.objective]()
        result = run_hifi_simulation(
            grid=grid, output_dir=Path(args.output_dir),
            config=config, objective=objective,
        )
        if result["success"]:
            print(f"\nFitness: {result['objective_score']['fitness']:.4f}")
            print(f"Time: {result['time_s']:.0f}s")
        else:
            print(f"\nFAILED: {result.get('error')}")
    elif args.results:
        evaluate_ga_results(
            results_path=Path(args.results),
            objective_name=args.objective,
            output_dir=Path(args.output_dir),
            top_k=args.top_k,
            config=config,
        )
    else:
        parser.error("Provide either --grid or --results")


if __name__ == "__main__":
    main()
