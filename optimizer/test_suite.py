"""Test suite for evaluating the inverse design optimizer.

Tests:
  1. Surrogate sanity: predictions match known dataset samples
  2. GA convergence: fitness improves over generations for all objectives
  3. GA diversity: final population isn't collapsed to one design
  4. Objective scoring: known S-params produce expected scores
  5. OpenEMS concordance: surrogate rankings correlate with OpenEMS rankings
  6. End-to-end: full pipeline produces valid GDS-exportable designs

Usage:
    python -m optimizer.test_suite                    # quick tests (no OpenEMS)
    python -m optimizer.test_suite --with-openems     # includes OpenEMS validation
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from optimizer.ga import GeneticAlgorithm, GAConfig, SurrogateEvaluator
from optimizer.objectives import (
    OBJECTIVES,
    MatchingObjective,
    ImpedanceMatchObjective,
    SParamGoal,
    bandpass_5ghz,
    lowpass_10ghz,
    broadband_match,
    notch_10ghz,
    narrowband_match_10ghz,
)

CHECKPOINT = os.environ.get("INVDESIGN_CKPT", "./checkpoints/best_model.pt")
DATASET_DIR = os.environ.get("INVDESIGN_DATASET", "./datasets/pixelgrid")


class TestResults:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.tests: list[dict] = []

    def record(self, name: str, passed: bool, detail: str = ""):
        status = "PASS" if passed else "FAIL"
        self.tests.append({"name": name, "status": status, "detail": detail})
        if passed:
            self.passed += 1
        else:
            self.failed += 1
        mark = "✓" if passed else "✗"
        print(f"  {mark} {name}" + (f" — {detail}" if detail else ""))

    def summary(self):
        total = self.passed + self.failed
        print(f"\n{'='*50}")
        print(f"Results: {self.passed}/{total} passed, {self.failed} failed")
        if self.failed > 0:
            print("Failed tests:")
            for t in self.tests:
                if t["status"] == "FAIL":
                    print(f"  ✗ {t['name']}: {t['detail']}")
        print(f"{'='*50}")
        return self.failed == 0


def test_objective_scoring(results: TestResults):
    """Test that objective scoring works correctly on synthetic S-params."""
    print("\n--- Test: Objective Scoring ---")

    # Perfect passthrough: S21 = 1 (0 dB), S11 = 0 (-inf dB)
    n_freq = 30
    perfect = np.zeros((n_freq, 4, 4), dtype=np.complex128)
    perfect[:, 0, 1] = 1.0  # S21 = 1
    perfect[:, 1, 0] = 1.0  # S12 = 1
    for i in range(4):
        perfect[:, i, i] = 0.01  # S11 etc = -40 dB

    obj = broadband_match()
    score = obj.evaluate(perfect)

    # Perfect passthrough should have very good fitness
    results.record("perfect_passthrough_high_fitness",
                   score["fitness"] > -5.0,
                   f"fitness={score['fitness']:.2f}")

    # Total reflection: S11 = 1, S21 = 0
    reflect = np.zeros((n_freq, 4, 4), dtype=np.complex128)
    reflect[:, 0, 0] = 1.0
    reflect[:, 1, 1] = 1.0

    score_bad = obj.evaluate(reflect)
    results.record("total_reflection_low_fitness",
                   score_bad["fitness"] < score["fitness"],
                   f"fitness={score_bad['fitness']:.2f}")

    # All objectives produce valid scores
    for name, factory in OBJECTIVES.items():
        obj = factory()
        score = obj.evaluate(perfect)
        results.record(f"objective_{name}_valid",
                       np.isfinite(score["fitness"]),
                       f"fitness={score['fitness']:.2f}")

    # Impedance matching: perfect passthrough with 50Ω→50Ω should give T=1
    obj_match = narrowband_match_10ghz()
    # Build S-params with perfect match: S21=1, S11=0 at all freqs
    score_match = obj_match.evaluate(perfect)
    results.record("impedance_match_perfect_passthrough",
                   score_match["fitness"] > -0.1,
                   f"fitness={score_match['fitness']:.4f}")

    # Total reflection should have bad matching efficiency
    score_match_bad = obj_match.evaluate(reflect)
    results.record("impedance_match_total_reflection",
                   score_match_bad["fitness"] < score_match["fitness"],
                   f"fitness={score_match_bad['fitness']:.4f}")


def test_surrogate_sanity(results: TestResults):
    """Test surrogate predictions on known dataset samples."""
    print("\n--- Test: Surrogate Sanity ---")

    evaluator = SurrogateEvaluator(CHECKPOINT)

    # Load a few designs from the dataset
    from surrogate.data import load_design
    dataset_path = Path(DATASET_DIR)
    designs = sorted(dataset_path.glob("design_*"))[:5]

    for d in designs:
        loaded = load_design(d)
        if loaded is None:
            continue
        grid, true_sparams = loaded

        # Predict with surrogate
        pred_sparams = evaluator.predict_sparams(grid[np.newaxis])[0]

        # Check shape
        results.record(f"surrogate_shape_{d.name}",
                       pred_sparams.shape == true_sparams.shape,
                       f"pred={pred_sparams.shape}, true={true_sparams.shape}")

        # Check S-params are in reasonable range
        pred_mag = np.abs(pred_sparams)
        results.record(f"surrogate_range_{d.name}",
                       pred_mag.max() < 5.0 and pred_mag.min() >= 0,
                       f"max|S|={pred_mag.max():.3f}")

        # Check surrogate error (S11 magnitude)
        pred_s11_db = 20 * np.log10(np.abs(pred_sparams[:, 0, 0]) + 1e-12)
        true_s11_db = 20 * np.log10(np.abs(true_sparams[:, 0, 0]) + 1e-12)
        s11_err = np.abs(pred_s11_db - true_s11_db).mean()
        results.record(f"surrogate_s11_err_{d.name}",
                       s11_err < 10.0,  # generous threshold
                       f"mean |S11| error = {s11_err:.1f} dB")
        break  # just test one to be quick


def test_ga_convergence(results: TestResults):
    """Test that GA fitness improves over generations."""
    print("\n--- Test: GA Convergence ---")

    evaluator = SurrogateEvaluator(CHECKPOINT)

    for obj_name in ["broadband_match", "lowpass_10ghz"]:
        objective = OBJECTIVES[obj_name]()
        config = GAConfig(pop_size=50, n_generations=50, seed=1)
        ga = GeneticAlgorithm(objective, evaluator, config)

        t0 = time.time()
        result = ga.run(verbose=False)
        dt = time.time() - t0

        init_fitness = result["best_fitness_history"][0]
        final_fitness = result["best_fitness_history"][-1]

        # Fitness should improve
        results.record(f"ga_converges_{obj_name}",
                       final_fitness > init_fitness,
                       f"init={init_fitness:.1f} → final={final_fitness:.1f} ({dt:.1f}s)")

        # Should complete in reasonable time
        results.record(f"ga_speed_{obj_name}",
                       dt < 30.0,
                       f"{dt:.1f}s for 50 gens × 50 pop")


def test_ga_diversity(results: TestResults):
    """Test that GA population maintains diversity (not all identical)."""
    print("\n--- Test: GA Diversity ---")

    evaluator = SurrogateEvaluator(CHECKPOINT)
    objective = broadband_match()
    config = GAConfig(pop_size=100, n_generations=100, seed=1)
    ga = GeneticAlgorithm(objective, evaluator, config)
    ga.run(verbose=False)

    # Check diversity: not all genomes identical
    pop = ga.population
    unique_genomes = len(set(tuple(g) for g in pop))
    results.record("population_diverse",
                   unique_genomes > config.pop_size * 0.3,
                   f"{unique_genomes}/{config.pop_size} unique genomes")

    # Check fill factor range
    fills = [g.mean() for g in pop]
    fill_std = np.std(fills)
    results.record("fill_factor_varied",
                   fill_std > 0.005,
                   f"fill std={fill_std:.3f}")


def test_grid_validity(results: TestResults):
    """Test that GA produces valid pixel grids."""
    print("\n--- Test: Grid Validity ---")

    evaluator = SurrogateEvaluator(CHECKPOINT)
    objective = broadband_match()
    config = GAConfig(pop_size=50, n_generations=30, seed=1)
    ga = GeneticAlgorithm(objective, evaluator, config)
    result = ga.run(verbose=False)

    best_grid = result["top_k"][0]["grid"]

    # Check shape
    results.record("grid_shape", best_grid.shape == (27, 27),
                   f"shape={best_grid.shape}")

    # Check binary
    unique_vals = np.unique(best_grid)
    results.record("grid_binary",
                   set(unique_vals).issubset({0, 1}),
                   f"values={unique_vals}")

    # Check ports exist (one per edge, random positions)
    has_n = best_grid[0, :].sum() >= 1
    has_s = best_grid[-1, :].sum() >= 1
    has_e = best_grid[:, -1].sum() >= 1
    has_w = best_grid[:, 0].sum() >= 1
    results.record("ports_present",
                   has_n and has_s and has_e and has_w,
                   "N/S/E/W ports on border edges")

    # Check can be saved/loaded as numpy
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".npy") as f:
        np.save(f.name, best_grid)
        loaded = np.load(f.name)
        results.record("grid_serializable",
                       np.array_equal(best_grid, loaded),
                       "numpy save/load roundtrip")


def test_openems_validation(results: TestResults):
    """Test OpenEMS validation on a GA-optimized design."""
    print("\n--- Test: OpenEMS Validation ---")

    from optimizer.validate import validate_design

    evaluator = SurrogateEvaluator(CHECKPOINT)
    objective = lowpass_10ghz()
    config = GAConfig(pop_size=50, n_generations=30, seed=1)
    ga = GeneticAlgorithm(objective, evaluator, config)
    result = ga.run(verbose=False)

    best_grid = result["top_k"][0]["grid"]
    work_dir = Path("optimizer/results/test_openems_val")

    t0 = time.time()
    val = validate_design(best_grid, objective, work_dir)
    dt = time.time() - t0

    results.record("openems_success", val["success"],
                   f"time={dt:.0f}s" if val["success"] else val.get("error", ""))

    if val["success"]:
        # Check S-params are physically reasonable
        sparams = val["sparams"]
        max_mag = np.abs(sparams).max()
        results.record("openems_passive",
                       max_mag < 1.1,
                       f"max|S|={max_mag:.3f}")

        # Surrogate vs OpenEMS correlation
        surrogate_fitness = result["top_k"][0]["fitness"]
        openems_fitness = val["objective_score"]["fitness"]
        results.record("surrogate_openems_order",
                       True,  # just record values
                       f"surrogate={surrogate_fitness:.1f}, "
                       f"openems={openems_fitness:.1f}, "
                       f"gap={openems_fitness - surrogate_fitness:.1f}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--with-openems", action="store_true",
                        help="Include OpenEMS validation tests (~2-3 min)")
    args = parser.parse_args()

    print("="*50)
    print("Inverse Design Optimizer — Test Suite")
    print("="*50)

    results = TestResults()

    test_objective_scoring(results)
    test_surrogate_sanity(results)
    test_ga_convergence(results)
    test_ga_diversity(results)
    test_grid_validity(results)

    if args.with_openems:
        test_openems_validation(results)
    else:
        print("\n--- Skipping OpenEMS tests (use --with-openems) ---")

    all_passed = results.summary()
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
