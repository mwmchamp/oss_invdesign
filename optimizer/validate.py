"""OpenEMS validation for top GA candidates."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from invdesign.simulate import SimConfig, simulate_grid, VALIDATION_CONFIG, HIFI_CONFIG
from optimizer.objectives import MatchingObjective


def validate_design(
    grid: np.ndarray,
    objective: MatchingObjective,
    work_dir: Path,
    *,
    config: SimConfig | None = None,
) -> dict:
    """Run FDTD on a candidate grid and score against objective."""
    config = config or VALIDATION_CONFIG
    result = simulate_grid(grid, work_dir, config)

    if result["success"]:
        result["objective_score"] = objective.evaluate(result["sparams"])

    return result


def validate_top_k(
    results: dict,
    objective: MatchingObjective,
    output_dir: Path,
    k: int = 3,
    high_fidelity: bool = False,
) -> list[dict]:
    """Validate top-K candidates from GA results with FDTD."""
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    config = HIFI_CONFIG if high_fidelity else VALIDATION_CONFIG
    validated = []

    for i, candidate in enumerate(results["top_k"][:k]):
        print(f"\nValidating candidate {i+1}/{k} "
              f"(surrogate fitness={candidate['fitness']:.4f})...")

        work_dir = output_dir / f"candidate_{i:02d}"
        val_result = validate_design(
            grid=candidate["grid"],
            objective=objective,
            work_dir=work_dir,
            config=config,
        )

        if val_result["success"]:
            surrogate_score = candidate["fitness"]
            openems_score = val_result["objective_score"]["fitness"]
            print(f"  Surrogate fitness: {surrogate_score:.4f}")
            print(f"  OpenEMS fitness:   {openems_score:.4f}")
            print(f"  Sim time: {val_result['time_s']:.0f}s")
            for g in val_result["objective_score"]["goals"]:
                print(f"    {g['goal']}: {g['achieved_db']:.1f}dB")
        else:
            print(f"  FAILED: {val_result.get('error', 'unknown')}")

        val_result["rank"] = i
        val_result["surrogate_fitness"] = candidate["fitness"]
        validated.append(val_result)

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
            entry["openems_fitness"] = v["objective_score"]["fitness"]
            entry["goals"] = v["objective_score"]["goals"]
        summary.append(entry)

    with (output_dir / "validation_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    return validated
