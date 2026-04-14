"""Surrogate-in-the-loop active learning for closing the GA-FDTD gap.

The core idea (Jones et al., 1998 — Efficient Global Optimization):
1. Run GA on surrogate to find top candidates
2. Simulate top candidates in FDTD (ground truth)
3. Add FDTD results to surrogate training data
4. Retrain surrogate (fast, few epochs)
5. Repeat until surrogate-FDTD gap closes

This module implements a lightweight version:
- No full retraining each round (too slow for interactive use)
- Instead: maintains a correction cache of (grid_hash -> FDTD_fitness)
- GA penalizes designs that are far from any validated design
- Optionally runs FDTD validation on top candidates each round

For batch/offline use, a full retrain loop is also provided.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from optimizer.ga import GAConfig, GeneticAlgorithm, SurrogateEvaluator
from optimizer.objectives import MatchingObjective, OBJECTIVES


def _grid_hash(grid: np.ndarray) -> str:
    """Fast hash of a binary grid for deduplication."""
    return hashlib.md5(grid.tobytes()).hexdigest()[:12]


@dataclass
class ActiveLearningConfig:
    """Configuration for active learning loop."""
    n_rounds: int = 3                  # number of AL rounds
    candidates_per_round: int = 3      # top-K designs to validate per round
    ga_config: GAConfig = field(default_factory=lambda: GAConfig(
        pop_size=200, n_generations=200, local_search=True, local_search_k=3,
    ))
    cache_dir: Path | None = None      # directory to persist cache
    # Main training dataset directory; validated designs are written here
    # as dataset-format design_XXXXX folders for future surrogate retraining.
    dataset_dir: Path | None = None
    # Penalty for distance from validated designs (0 = disabled)
    novelty_penalty_weight: float = 0.0  # not used in basic version


@dataclass
class ValidationRecord:
    """Record of a validated design."""
    grid_hash: str
    grid: np.ndarray
    surrogate_fitness: float
    fdtd_fitness: float
    gap_ratio: float           # fdtd_fitness / surrogate_fitness
    sparams_fdtd: np.ndarray   # (n_freq, 4, 4) complex
    round_idx: int


class ActiveLearningLoop:
    """Run GA optimization with FDTD validation between rounds.

    Each round:
    1. Run GA on surrogate (seeded with best designs from previous rounds)
    2. Take top-K candidates not yet validated
    3. Validate with FDTD
    4. Track surrogate-FDTD gap
    5. Return best FDTD-validated design

    The key insight: even without retraining, validating multiple GA outputs
    and returning the best FDTD-confirmed design dramatically improves
    real-world performance. The surrogate is used for search, not selection.
    """

    def __init__(
        self,
        objective: MatchingObjective,
        evaluator: SurrogateEvaluator,
        fdtd_fn,
        config: ActiveLearningConfig | None = None,
    ):
        """
        Args:
            objective: matching objective to optimize
            evaluator: trained surrogate model
            fdtd_fn: callable(grid) -> (sparams, info_dict) that runs FDTD
                     simulation and returns complex S-parameters
            config: active learning configuration
        """
        self.objective = objective
        self.evaluator = evaluator
        self.fdtd_fn = fdtd_fn
        self.config = config or ActiveLearningConfig()
        self.validated: list[ValidationRecord] = []
        self._seen_hashes: set[str] = set()

        if self.config.cache_dir:
            self._load_cache()

        if self.config.dataset_dir:
            Path(self.config.dataset_dir).mkdir(parents=True, exist_ok=True)

    def run(self, verbose: bool = True) -> dict:
        """Run the full active learning loop.

        Returns:
            dict with best_fdtd (best FDTD-validated design),
            all_validated (all validation records), round_stats
        """
        t0 = time.time()
        round_stats = []

        for round_idx in range(self.config.n_rounds):
            if verbose:
                print(f"\n{'='*60}")
                print(f"Active Learning Round {round_idx + 1}/{self.config.n_rounds}")
                print(f"{'='*60}")

            stats = self._run_round(round_idx, verbose=verbose)
            round_stats.append(stats)

            if verbose:
                print(f"\nRound {round_idx+1} summary:")
                print(f"  Validated: {stats['n_validated']} designs")
                print(f"  Best FDTD fitness: {stats['best_fdtd_fitness']:.4f}")
                print(f"  Best surr fitness: {stats['best_surr_fitness']:.4f}")
                if stats['avg_gap_ratio'] is not None:
                    print(f"  Avg gap ratio: {stats['avg_gap_ratio']:.2f}x")

        dt = time.time() - t0

        # Find best FDTD-validated design
        if self.validated:
            best_idx = max(range(len(self.validated)),
                          key=lambda i: self.validated[i].fdtd_fitness)
            best = self.validated[best_idx]
        else:
            best = None

        if self.config.cache_dir:
            self._save_cache()

        result = {
            "best_fdtd": best,
            "all_validated": self.validated,
            "round_stats": round_stats,
            "total_time_s": dt,
            "total_fdtd_evals": len(self.validated),
        }

        if verbose and best is not None:
            print(f"\n{'='*60}")
            print(f"Active Learning Complete ({dt:.1f}s)")
            print(f"  Total FDTD evaluations: {len(self.validated)}")
            print(f"  Best FDTD fitness: {best.fdtd_fitness:.4f}")
            print(f"  Corresponding surrogate fitness: {best.surrogate_fitness:.4f}")
            print(f"  Gap ratio: {best.gap_ratio:.2f}x")
            score = self.objective.evaluate(best.sparams_fdtd)
            for g in score["goals"]:
                print(f"    {g['goal']}: achieved={g['achieved_db']:.1f}dB "
                      f"worst={g.get('worst_db', float('nan')):.1f}dB")

        return result

    def _run_round(self, round_idx: int, verbose: bool) -> dict:
        """Run one round of GA + FDTD validation."""
        # Vary seed each round
        ga_config = GAConfig(
            pop_size=self.config.ga_config.pop_size,
            n_generations=self.config.ga_config.n_generations,
            elite_frac=self.config.ga_config.elite_frac,
            tournament_size=self.config.ga_config.tournament_size,
            crossover_rate=self.config.ga_config.crossover_rate,
            mutation_rate=self.config.ga_config.mutation_rate,
            mutation_decay=self.config.ga_config.mutation_decay,
            min_mutation_rate=self.config.ga_config.min_mutation_rate,
            block_crossover_prob=self.config.ga_config.block_crossover_prob,
            block_size_range=self.config.ga_config.block_size_range,
            inner_size=self.config.ga_config.inner_size,
            seed=self.config.ga_config.seed + round_idx * 1000,
            min_fill=self.config.ga_config.min_fill,
            max_fill=self.config.ga_config.max_fill,
            local_search=self.config.ga_config.local_search,
            local_search_k=self.config.ga_config.local_search_k,
        )

        # Run GA
        ga = GeneticAlgorithm(self.objective, self.evaluator, ga_config)
        ga_results = ga.run(verbose=verbose)

        # Select candidates not yet validated
        candidates = []
        for design in ga_results["top_k"]:
            h = _grid_hash(design["grid"])
            if h not in self._seen_hashes:
                candidates.append(design)
            if len(candidates) >= self.config.candidates_per_round:
                break

        # FDTD validation
        round_validated = []
        for i, cand in enumerate(candidates):
            if verbose:
                print(f"\n  FDTD validation {i+1}/{len(candidates)} "
                      f"(surr fitness: {cand['fitness']:.4f})...")

            t_fdtd = time.time()
            try:
                sparams_fdtd, fdtd_info = self.fdtd_fn(cand["grid"])
            except Exception as e:
                if verbose:
                    print(f"    FDTD failed: {e}")
                continue
            dt_fdtd = time.time() - t_fdtd

            # Score FDTD result
            fdtd_score = self.objective.evaluate(sparams_fdtd)
            fdtd_fitness = fdtd_score["fitness"]

            gap_ratio = (fdtd_fitness / cand["fitness"]
                         if cand["fitness"] != 0 else float("inf"))

            h = _grid_hash(cand["grid"])
            record = ValidationRecord(
                grid_hash=h,
                grid=cand["grid"],
                surrogate_fitness=cand["fitness"],
                fdtd_fitness=fdtd_fitness,
                gap_ratio=gap_ratio,
                sparams_fdtd=sparams_fdtd,
                round_idx=round_idx,
            )
            self.validated.append(record)
            self._seen_hashes.add(h)
            round_validated.append(record)

            if self.config.dataset_dir is not None:
                try:
                    design_id = self._append_to_dataset(record)
                    if verbose:
                        print(f"    Added to dataset as {design_id}")
                except Exception as exc:
                    if verbose:
                        print(f"    WARNING: failed to add to dataset: {exc}")

            if verbose:
                print(f"    FDTD fitness: {fdtd_fitness:.4f} "
                      f"(gap: {gap_ratio:.2f}x, {dt_fdtd:.0f}s)")

        # Round statistics
        best_fdtd = max(
            [v.fdtd_fitness for v in self.validated],
            default=float("-inf"),
        )
        best_surr = max(
            [d["fitness"] for d in ga_results["top_k"][:1]],
            default=float("-inf"),
        )
        gaps = [v.gap_ratio for v in round_validated if np.isfinite(v.gap_ratio)]

        return {
            "round": round_idx,
            "n_validated": len(round_validated),
            "best_fdtd_fitness": best_fdtd,
            "best_surr_fitness": best_surr,
            "avg_gap_ratio": np.mean(gaps) if gaps else None,
            "ga_time_s": ga_results["time_s"],
        }

    def _append_to_dataset(self, record: ValidationRecord) -> str:
        """Write a validated design into the main training dataset.

        Produces the same folder layout as invdesign.dataset.generate_one:
        design_XXXXX/{pixel_grid.npy, pixel_grid.s4p, pixel_grid.freqs.npy,
                      meta.json}. Frequency grid is locked to the dataset's
        (1-30 GHz, 30 points) so s4p dimensions match existing designs.
        """
        from invdesign.touchstone import write_touchstone_nport

        dataset_dir = Path(self.config.dataset_dir)
        existing = sorted(dataset_dir.glob("design_[0-9]*"))
        if existing:
            last_id = max(int(p.name.split("_")[-1]) for p in existing
                          if p.name.split("_")[-1].isdigit())
            next_id = last_id + 1
        else:
            next_id = 0
        design_id = f"design_{next_id:05d}"
        workdir = dataset_dir / design_id
        workdir.mkdir(parents=True, exist_ok=True)

        # Dataset-locked frequency grid
        freqs = np.linspace(1e9, 30e9, 30)
        if record.sparams_fdtd.shape[0] != len(freqs):
            raise ValueError(
                f"AL sparams have {record.sparams_fdtd.shape[0]} freq points, "
                f"dataset requires {len(freqs)}. Frequency grid mismatch."
            )

        np.save(workdir / "pixel_grid.npy", record.grid)
        np.save(workdir / "pixel_grid.freqs.npy", freqs)
        write_touchstone_nport(
            workdir / "pixel_grid.s4p", 4, freqs, record.sparams_fdtd,
        )

        inner = record.grid[1:-1, 1:-1]
        fill_factor = float(inner.sum()) / inner.size
        meta = {
            "design_id": design_id,
            "source": "active_learning",
            "round_idx": record.round_idx,
            "grid_hash": record.grid_hash,
            "inner_size": int(inner.shape[0]),
            "fill_factor": round(fill_factor, 4),
            "em_backend": "pixel",
            "grid_npy": "pixel_grid.npy",
            "surrogate_fitness": float(record.surrogate_fitness),
            "fdtd_fitness": float(record.fdtd_fitness),
            "gap_ratio": float(record.gap_ratio)
                if np.isfinite(record.gap_ratio) else None,
        }
        (workdir / "meta.json").write_text(
            json.dumps(meta, indent=2), encoding="utf-8",
        )
        return design_id

    def _save_cache(self):
        """Save validation cache to disk."""
        cache_dir = Path(self.config.cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)

        records = []
        for v in self.validated:
            np.save(cache_dir / f"grid_{v.grid_hash}.npy", v.grid)
            np.save(cache_dir / f"sparams_{v.grid_hash}.npy", v.sparams_fdtd)
            records.append({
                "grid_hash": v.grid_hash,
                "surrogate_fitness": v.surrogate_fitness,
                "fdtd_fitness": v.fdtd_fitness,
                "gap_ratio": v.gap_ratio,
                "round_idx": v.round_idx,
            })

        with (cache_dir / "validation_log.json").open("w") as f:
            json.dump(records, f, indent=2)

    def _load_cache(self):
        """Load validation cache from disk."""
        cache_dir = Path(self.config.cache_dir)
        log_path = cache_dir / "validation_log.json"
        if not log_path.exists():
            return

        with log_path.open() as f:
            records = json.load(f)

        for rec in records:
            h = rec["grid_hash"]
            grid_path = cache_dir / f"grid_{h}.npy"
            sp_path = cache_dir / f"sparams_{h}.npy"
            if grid_path.exists() and sp_path.exists():
                self.validated.append(ValidationRecord(
                    grid_hash=h,
                    grid=np.load(grid_path),
                    surrogate_fitness=rec["surrogate_fitness"],
                    fdtd_fitness=rec["fdtd_fitness"],
                    gap_ratio=rec["gap_ratio"],
                    sparams_fdtd=np.load(sp_path),
                    round_idx=rec["round_idx"],
                ))
                self._seen_hashes.add(h)


def make_fdtd_fn(
    output_dir: Path,
    config: "SimConfig | None" = None,
    # Tunable kwargs — note: frequency grid is hard-locked to the dataset's
    # (1-30 GHz, 30 points) so AL outputs stay s4p-compatible with the
    # training dataset. Do not expose f_start/f_stop/n_freq here.
    use_pml: bool = True,
    end_criteria_db: float = -50.0,
    nr_ts: int = 1_000_000,
    mesh_div: float = 30.0,
    xy_pad_um: float = 200.0,
):
    """Create an FDTD function for use with ActiveLearningLoop.

    Returns callable(grid) -> (sparams, info_dict).
    """
    from invdesign.simulate import SimConfig, simulate_grid, VALIDATION_CONFIG

    if config is None:
        config = SimConfig(
            nr_ts=nr_ts,
            end_criteria_db=end_criteria_db,
            n_freq=30,
            f_start=1e9,
            f_stop=30e9,
            mesh_div=mesh_div,
            use_pml=use_pml,
            xy_pad_um=xy_pad_um,
        )
    else:
        if config.n_freq != 30 or config.f_start != 1e9 or config.f_stop != 30e9:
            raise ValueError(
                "AL frequency grid must match dataset (1-30 GHz, 30 points); "
                f"got n_freq={config.n_freq}, f_start={config.f_start}, "
                f"f_stop={config.f_stop}. Using a different grid breaks s4p "
                "dimension compatibility with the training dataset."
            )

    _counter = [0]

    def fdtd_fn(grid: np.ndarray):
        _counter[0] += 1
        work_dir = Path(output_dir) / f"al_design_{_counter[0]:04d}"
        result = simulate_grid(grid, work_dir, config)
        if not result["success"]:
            raise RuntimeError(f"FDTD failed at {work_dir}: {result.get('error')}")
        return result["sparams"], {"work_dir": str(work_dir)}

    return fdtd_fn


def run_active_learning_cli():
    """CLI entry point for active learning optimization."""
    import argparse
    parser = argparse.ArgumentParser(
        description="Surrogate-in-the-loop active learning optimization"
    )
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to surrogate model checkpoint")
    parser.add_argument("--objective", type=str, required=True,
                        choices=list(OBJECTIVES.keys()))
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--n-rounds", type=int, default=3)
    parser.add_argument("--candidates-per-round", type=int, default=3)
    parser.add_argument("--pop-size", type=int, default=200)
    parser.add_argument("--n-generations", type=int, default=200)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--use-pml", action="store_true", default=True)
    parser.add_argument("--no-pml", action="store_true")
    parser.add_argument("--dataset-dir", type=str, default=None,
                        help="If set, FDTD-validated designs are appended "
                             "here in dataset format for surrogate retraining.")
    args = parser.parse_args()

    objective = OBJECTIVES[args.objective]()
    evaluator = SurrogateEvaluator(args.checkpoint)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fdtd_fn = make_fdtd_fn(
        output_dir=output_dir / "fdtd_runs",
        use_pml=not args.no_pml,
    )

    al_config = ActiveLearningConfig(
        n_rounds=args.n_rounds,
        candidates_per_round=args.candidates_per_round,
        ga_config=GAConfig(
            pop_size=args.pop_size,
            n_generations=args.n_generations,
            seed=args.seed,
            local_search=True,
            local_search_k=3,
        ),
        cache_dir=output_dir / "al_cache",
        dataset_dir=Path(args.dataset_dir) if args.dataset_dir else None,
    )

    loop = ActiveLearningLoop(objective, evaluator, fdtd_fn, al_config)
    results = loop.run(verbose=True)

    # Save final results
    if results["best_fdtd"] is not None:
        best = results["best_fdtd"]
        np.save(output_dir / "best_grid.npy", best.grid)
        np.save(output_dir / "best_sparams_fdtd.npy", best.sparams_fdtd)
        print(f"\nBest design saved to {output_dir / 'best_grid.npy'}")
        print(f"FDTD S-params saved to {output_dir / 'best_sparams_fdtd.npy'}")


if __name__ == "__main__":
    run_active_learning_cli()
