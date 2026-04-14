"""Genetic algorithm for pixel grid inverse design.

Optimizes the 25×25 inner binary pixel grid to achieve target S-parameters,
using the trained CNN surrogate for fast fitness evaluation.

Key design decisions:
  - Binary genome: 625 bits (25×25 inner pixels)
  - Port positions: one per edge, evolved with Gaussian jitter (σ=0.5)
  - Crossover: 50/50 row-wise or column-wise (per-line cut-point) [default],
    with legacy modes still available
  - Mutation: bit-flip with linear decay (rate - gen/n_gen * 0.05) [default]
  - Selection: tournament with elitism
  - Fitness: surrogate-predicted S-params scored against MatchingObjective
  - SVD passivity enforcement on predicted S-parameters

Reference: EM300 / JSSC 2023 GA-based matching network synthesis.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import torch

from surrogate.model import SParamCNN, SParamCNNv2, SParamResNet, SParamEM300
from optimizer.objectives import MatchingObjective
from invdesign.grid_gen import generate_connected_grid, place_ports, _pick_port_positions

# Upper-triangle indices for vectorized reconstruction
_UTRI_ROWS, _UTRI_COLS = np.triu_indices(4)


@dataclass
class GAConfig:
    """Genetic algorithm hyperparameters.

    Defaults follow Karahan et al.\\ (Nat.\\ Commun.\\ 2024): M=4096 population,
    tournament = M/16, mutation 0.1 → 0 linearly over 100 generations, 8
    elites carried forward. We run at M=1024 (quarter-scale) since surrogate
    evaluation is batched on GPU but not free, keeping the paper's 8 elites,
    and add our own extensions (path-seeded init, port co-evolution, memetic
    single-pixel local search).
    """
    pop_size: int = 1024
    n_generations: int = 100
    elite_frac: float = 8 / 1024         # default 8 elites at pop_size=1024 (Karahan Nat. Commun. 2024)
    tournament_size: int = 64            # M/16, matches paper's 256/4096 ratio
    crossover_rate: float = 0.8
    mutation_rate: float = 0.1           # paper's initial rate
    mutation_decay: float = 0.995        # per-generation multiplicative decay (legacy)
    min_mutation_rate: float = 0.0       # paper anneals to zero (pure exploitation)
    block_crossover_prob: float = 0.3    # probability of using block crossover (legacy)
    block_size_range: tuple[int, int] = (3, 10)
    inner_size: int = 25
    seed: int = 1
    # Fill factor constraints (optional)
    min_fill: float = 0.05
    max_fill: float = 0.95
    local_search: bool = True            # enable greedy single-pixel hill-climb
    local_search_k: int = 5              # number of top designs to refine
    # Per-pixel penalty on "stranded" metal (pixels not reachable via
    # 4-connectivity from any active-pair port). Disabled by default:
    # empirically, an active stranded penalty pushes the GA toward very
    # dense blobs (connecting noise rather than deleting it). Leaving the
    # machinery available but off; set >0 to experiment.
    stranded_metal_penalty: float = 0.0
    # ── Reference-style operators (EM300 / JSSC 2023 / Nat. Commun. 2024) ──
    crossover_mode: str = "rowcol"       # "rowcol" (50/50 row/col) | "row" | "col" | "mixed"
    mutation_decay_mode: str = "linear"  # "linear" | "multiplicative" (legacy)
    linear_decay_delta: float = 0.1      # mutation_rate - gen/n_gen * delta (→ 0 at end)
    port_jitter_sigma: float = 0.5       # Gaussian σ for port position jitter (pixels)
    # DC-path feasibility penalty (Karahan JSSC 2023): penalize grids that
    # lack a metal path between coupled port pairs. Heavy additive penalty
    # applied to fitness before selection. Set to 0 to disable.
    dc_path_penalty: float = 200.0


class SurrogateEvaluator:
    """Batch-evaluate pixel grids using the trained CNN surrogate."""

    def __init__(
        self,
        checkpoint_path: str,
        device: str | None = None,
    ):
        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Limit CPU threads to avoid contention on multi-core nodes
        if self.device.type == "cpu" and torch.get_num_threads() > 8:
            torch.set_num_threads(4)

        # Load model and normalization stats
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        args = ckpt.get("args", {})

        model_version = args.get("model_version", "v1")
        model_kwargs = dict(
            n_freq=args.get("n_freq", 30),
            n_utri=10,
            base_channels=args.get("base_channels", 64),
        )
        if model_version == "v4":
            self.model = SParamEM300(
                dropout=args.get("dropout", 0.15),
                output_activation=args.get("output_activation", "tanh"),
                **model_kwargs,
            ).to(self.device)
        elif model_version == "v3":
            self.model = SParamResNet(**model_kwargs).to(self.device)
        elif model_version == "v2":
            self.model = SParamCNNv2(**model_kwargs).to(self.device)
        else:
            self.model = SParamCNN(**model_kwargs).to(self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()

        self.target_mean = torch.from_numpy(ckpt["target_mean"]).to(self.device)
        self.target_std = torch.from_numpy(ckpt["target_std"]).to(self.device)

    def _raw_predict(self, grids: np.ndarray) -> np.ndarray:
        """Predict raw upper-tri S-params. Returns (N, F, 10, 2)."""
        x = torch.from_numpy(grids.astype(np.float32)).unsqueeze(1).to(self.device)
        pred_norm = self.model(x)
        pred_raw = pred_norm * self.target_std + self.target_mean
        return pred_raw.cpu().numpy()

    @torch.no_grad()
    def predict_sparams(self, grids: np.ndarray) -> np.ndarray:
        """Predict S-parameters for a batch of grids.

        Applies SVD-based passivity enforcement: clamps singular values
        of each S-matrix to ≤ 1, ensuring physical realizability.

        Args:
            grids: (N, 27, 27) binary grids

        Returns:
            (N, n_freq, 4, 4) complex S-parameters (passive)
        """
        pred_np = self._raw_predict(grids)
        sparams = _utri_to_full(pred_np)
        return _enforce_passivity_svd(sparams)


def _utri_to_full(pred_np: np.ndarray) -> np.ndarray:
    """Convert (N, F, 10, 2) upper-tri predictions to (N, F, 4, 4) complex."""
    N, F = pred_np.shape[:2]
    cpx = pred_np[..., 0] + 1j * pred_np[..., 1]  # (N, F, 10)
    sparams = np.zeros((N, F, 4, 4), dtype=np.complex128)
    sparams[:, :, _UTRI_ROWS, _UTRI_COLS] = cpx
    sparams[:, :, _UTRI_COLS, _UTRI_ROWS] = cpx
    return sparams


_PORT_EDGE_COORDS = {  # (edge_idx) → (row, col) in outer grid given port_pos (1..inner)
    0: lambda p, outer: (0, p),           # N: top row
    1: lambda p, outer: (outer - 1, p),   # S: bottom row
    2: lambda p, outer: (p, outer - 1),   # E: right col
    3: lambda p, outer: (p, 0),           # W: left col
}


def _grid_connected(grid: np.ndarray, r0: int, c0: int, r1: int, c1: int) -> bool:
    """Flood-fill (4-connected) on metal pixels; return True if (r1,c1) reached.

    Uses 4-connectivity (orthogonal neighbours only) because diagonal
    pixel-corner touches carry negligible current in FDTD simulation — an
    8-connected chain that threads the grid diagonally is electrically open,
    even though it is topologically connected. Using 4-connectivity here
    keeps the GA's DC-path penalty aligned with physical conduction.
    """
    H, W = grid.shape
    if grid[r0, c0] == 0 or grid[r1, c1] == 0:
        return False
    stack = [(r0, c0)]
    seen = np.zeros_like(grid, dtype=bool)
    seen[r0, c0] = True
    while stack:
        r, c = stack.pop()
        if r == r1 and c == c1:
            return True
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W and not seen[nr, nc] and grid[nr, nc] == 1:
                seen[nr, nc] = True
                stack.append((nr, nc))
    return False


def _enforce_passivity_svd(sparams: np.ndarray) -> np.ndarray:
    """Enforce passivity by clamping S-matrix singular values to ≤ 1.

    Per-element |S_ij| ≤ 1 is necessary but not sufficient for passivity.
    The correct constraint is that all singular values of S are ≤ 1 at every
    frequency, i.e. S^H S ⪯ I (no power gain). This is the standard approach
    used in microwave circuit simulation (cf. ga_matching_network reference).

    Args:
        sparams: (N, F, P, P) complex S-parameters
    Returns:
        (N, F, P, P) passive S-parameters
    """
    N, F, P, _ = sparams.shape
    out = sparams.copy()
    for n in range(N):
        for f in range(F):
            U, s, Vh = np.linalg.svd(out[n, f])
            if np.any(s > 1.0):
                s_clamped = np.minimum(s, 1.0)
                out[n, f] = U @ np.diag(s_clamped) @ Vh
    return out


class EnsembleEvaluator:
    """Ensemble of surrogates with uncertainty-penalized predictions.

    Loads N checkpoints, averages their predictions, and provides a
    disagreement metric. During GA, the optimizer penalizes designs
    where the ensemble disagrees — preventing exploitation of any
    single model's errors.

    Reference: Lakshminarayanan et al., "Simple and Scalable Predictive
    Uncertainty Estimation using Deep Ensembles" (NeurIPS 2017).
    """

    def __init__(
        self,
        checkpoint_paths: list[str],
        device: str | None = None,
        uncertainty_weight: float = 2.0,
    ):
        if not checkpoint_paths:
            raise ValueError("Need at least one checkpoint")

        self.members = [SurrogateEvaluator(p, device=device) for p in checkpoint_paths]
        self.device = self.members[0].device
        self.uncertainty_weight = uncertainty_weight

    @torch.no_grad()
    def predict_sparams(self, grids: np.ndarray) -> np.ndarray:
        """Predict mean S-parameters across ensemble members (with SVD passivity)."""
        preds = [m._raw_predict(grids) for m in self.members]
        mean_pred = np.mean(preds, axis=0)
        return _enforce_passivity_svd(_utri_to_full(mean_pred))

    @torch.no_grad()
    def predict_with_uncertainty(self, grids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Predict S-params and per-design uncertainty.

        Returns:
            sparams: (N, F, 4, 4) complex mean S-parameters
            uncertainty: (N,) scalar disagreement per design
                         (mean std of |S| across ensemble members)
        """
        # Each member: (N, F, 10, 2)
        preds = np.stack([m._raw_predict(grids) for m in self.members])  # (M, N, F, 10, 2)

        # Mean prediction
        mean_pred = preds.mean(axis=0)  # (N, F, 10, 2)
        sparams = _enforce_passivity_svd(_utri_to_full(mean_pred))

        # Uncertainty: std of |S| across members, averaged over freq and ports
        mags = np.sqrt(preds[..., 0] ** 2 + preds[..., 1] ** 2)  # (M, N, F, 10)
        std_mags = mags.std(axis=0)  # (N, F, 10)
        uncertainty = std_mags.mean(axis=(1, 2))  # (N,)

        return sparams, uncertainty


class GeneticAlgorithm:
    """GA optimizer for pixel grid inverse design."""

    def __init__(
        self,
        objective: MatchingObjective,
        evaluator: SurrogateEvaluator,
        config: GAConfig | None = None,
    ):
        self.objective = objective
        self.evaluator = evaluator
        self.config = config or GAConfig()
        self.rng = np.random.default_rng(self.config.seed)

        self.inner = self.config.inner_size
        self.outer = self.inner + 2
        self.n_bits = self.inner * self.inner

        # State
        self.population: np.ndarray | None = None  # (pop_size, n_bits)
        self.port_positions: np.ndarray | None = None  # (pop_size, 4) port positions
        self.fitness: np.ndarray | None = None
        self.best_fitness_history: list[float] = []
        self.mean_fitness_history: list[float] = []
        self.generation = 0
        self.mutation_rate = self.config.mutation_rate

    def _make_grid(self, genome: np.ndarray, ports: np.ndarray | None = None) -> np.ndarray:
        """Convert flat genome (625,) to full grid (27, 27) with port border.

        Parameters
        ----------
        genome : (n_bits,) binary array of inner pixels
        ports : (4,) array of port positions [N_col, S_col, E_row, W_row],
                each in [1, inner_size]. If None, uses centered ports as fallback.
        """
        grid = np.zeros((self.outer, self.outer), dtype=np.int8)
        grid[1:-1, 1:-1] = genome.reshape(self.inner, self.inner)
        if ports is None:
            mid = self.outer // 2
            ports = np.array([mid, mid, mid, mid])
        port_dict = {0: int(ports[0]), 1: int(ports[1]),
                     2: int(ports[2]), 3: int(ports[3])}
        place_ports(grid, port_dict)
        return grid

    def _get_active_port_pairs(self) -> list[tuple[int, int]]:
        """Extract port pairs from objective goals that need coupling (i != j)."""
        from optimizer.objectives import ImpedanceMatchObjective, ImpedanceMatchGoal
        pairs = set()

        if isinstance(self.objective, ImpedanceMatchObjective):
            # Impedance match goals define in_port → out_port
            for g in self.objective.goals:
                p1, p2 = min(g.in_port, g.out_port), max(g.in_port, g.out_port)
                pairs.add((p1, p2))
            # Also check sparam_goals
            for g in self.objective.sparam_goals:
                if g.i != g.j and g.mode in ("above", "at"):
                    pairs.add((min(g.i, g.j), max(g.i, g.j)))
        else:
            for g in self.objective.goals:
                if g.i != g.j and g.mode in ("above", "at"):
                    pairs.add((min(g.i, g.j), max(g.i, g.j)))

        return list(pairs) if pairs else [(0, 1)]

    def _init_population(self) -> None:
        """Initialize population: mix of path-seeded and random grids.

        Path-seeded grids (60%) have guaranteed metal paths between the
        target ports, giving the GA a head start on coupling. Random grids
        (40%) maintain diversity for exploration.

        Each individual also gets random port positions (one per edge).
        """
        pop = np.zeros((self.config.pop_size, self.n_bits), dtype=np.int8)
        ports = np.zeros((self.config.pop_size, 4), dtype=np.int32)
        port_pairs = self._get_active_port_pairs()
        n_seeded = int(self.config.pop_size * 0.6)

        for i in range(self.config.pop_size):
            # Random port positions for each individual
            pp = _pick_port_positions(self.rng, self.inner)
            ports[i] = [pp[0], pp[1], pp[2], pp[3]]

            if i < n_seeded:
                # Path-seeded: random walk waveguide connecting target ports
                seed_val = self.config.seed * 10000 + i
                pw = self.rng.integers(1, 3)
                grid = generate_connected_grid(
                    seed=seed_val,
                    inner_size=self.inner,
                    port_pairs=port_pairs,
                    path_width=pw,
                    fill_range=(self.config.min_fill, self.config.max_fill),
                )
                pop[i] = grid[1:-1, 1:-1].flatten()
                # Extract port positions from the connected grid's border
                for edge in range(4):
                    if edge == 0:  # N
                        border_cols = np.where(grid[0, 1:-1] == 1)[0] + 1
                    elif edge == 1:  # S
                        border_cols = np.where(grid[-1, 1:-1] == 1)[0] + 1
                    elif edge == 2:  # E
                        border_rows = np.where(grid[1:-1, -1] == 1)[0] + 1
                    elif edge == 3:  # W
                        border_rows = np.where(grid[1:-1, 0] == 1)[0] + 1
                    if edge < 2 and len(border_cols) > 0:
                        ports[i, edge] = int(border_cols[len(border_cols) // 2])
                    elif edge >= 2 and len(border_rows) > 0:
                        ports[i, edge] = int(border_rows[len(border_rows) // 2])
            else:
                # Random: varied fill for diversity
                fill = self.rng.uniform(self.config.min_fill, self.config.max_fill)
                pop[i] = (self.rng.random(self.n_bits) < fill).astype(np.int8)
        self.population = pop
        self.port_positions = ports

    def _stranded_metal_penalty(
        self, grids: np.ndarray, port_positions: np.ndarray,
    ) -> np.ndarray:
        """Per-design count of metal pixels not 4-connected to any active port.

        A 4-connected flood-fill is started from each active-pair port pixel;
        the union of visited metal forms the ``useful'' trace. Every metal
        pixel outside this union is stranded — it contributes fill% but no
        conduction, and visually appears as speckle on the exported layout.
        """
        w = self.config.stranded_metal_penalty
        if w <= 0:
            return np.zeros(len(grids))
        pairs = self._get_active_port_pairs()
        active_ports = sorted({p for pair in pairs for p in pair}) or [0, 1]
        outer = self.outer
        penalty = np.zeros(len(grids))
        for i, (grid, ports) in enumerate(zip(grids, port_positions)):
            reached = np.zeros_like(grid, dtype=bool)
            for p in active_ports:
                r0, c0 = _PORT_EDGE_COORDS[p](int(ports[p]), outer)
                if grid[r0, c0] == 0 or reached[r0, c0]:
                    continue
                stack = [(r0, c0)]
                reached[r0, c0] = True
                while stack:
                    r, c = stack.pop()
                    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                        nr, nc = r + dr, c + dc
                        if (0 <= nr < outer and 0 <= nc < outer
                                and not reached[nr, nc] and grid[nr, nc] == 1):
                            reached[nr, nc] = True
                            stack.append((nr, nc))
            n_stranded = int((grid == 1).sum() - reached.sum())
            penalty[i] = w * n_stranded
        return penalty

    def _dc_path_penalty(self, grids: np.ndarray, port_positions: np.ndarray) -> np.ndarray:
        """Additive fitness penalty summed over disconnected *active* port pairs.

        A port pair (a, b) is "active" when the objective explicitly requests
        coupling between them: either an ImpedanceMatchGoal (in_port, out_port)
        or an S-parameter goal with i != j and mode in {"above", "at"} (i.e.,
        a transmission target, not an isolation target). Active pairs are
        returned by `_get_active_port_pairs`.

        For each design we BFS from the port-a pixel and add `w` for every
        active pair whose metal path does not reach port b. Karahan et al.
        JSSC 2023 use the same per-pair formulation; isolation goals are
        intentionally excluded since disconnected pairs already help them.
        """
        w = self.config.dc_path_penalty
        if w <= 0:
            return np.zeros(len(grids))
        pairs = self._get_active_port_pairs()
        outer = self.outer
        penalty = np.zeros(len(grids))
        for i, (grid, ports) in enumerate(zip(grids, port_positions)):
            n_disconnected = 0
            for a, b in pairs:
                ra, ca = _PORT_EDGE_COORDS[a](int(ports[a]), outer)
                rb, cb = _PORT_EDGE_COORDS[b](int(ports[b]), outer)
                if not _grid_connected(grid, ra, ca, rb, cb):
                    n_disconnected += 1
            penalty[i] = w * n_disconnected
        return penalty

    def _evaluate_population(self) -> np.ndarray:
        """Evaluate fitness for entire population using surrogate.

        If the evaluator is an EnsembleEvaluator, applies uncertainty penalty:
        fitness_adjusted = fitness_raw - weight * uncertainty
        This prevents the GA from exploiting designs where the ensemble disagrees.

        Additionally applies a DC-path feasibility penalty (Karahan JSSC 2023)
        that subtracts a heavy constant per disconnected coupled-port pair.
        """
        grids = np.array([self._make_grid(g, p) for g, p in zip(self.population, self.port_positions)])

        use_ensemble = isinstance(self.evaluator, EnsembleEvaluator)
        if use_ensemble:
            sparams, uncertainty = self.evaluator.predict_with_uncertainty(grids)
        else:
            sparams = self.evaluator.predict_sparams(grids)

        fitness = np.zeros(self.config.pop_size)
        for i in range(self.config.pop_size):
            result = self.objective.evaluate(sparams[i])
            fitness[i] = result["fitness"]

        if use_ensemble:
            # Penalize uncertain designs (fitness is negative, penalty makes it more negative)
            penalty = self.evaluator.uncertainty_weight * uncertainty
            fitness -= penalty

        dc_pen = self._dc_path_penalty(grids, self.port_positions)
        fitness -= dc_pen

        strand_pen = self._stranded_metal_penalty(grids, self.port_positions)
        fitness -= strand_pen

        return fitness

    def _tournament_select(self) -> int:
        """Tournament selection — return index of winner."""
        candidates = self.rng.choice(
            self.config.pop_size, size=self.config.tournament_size, replace=False
        )
        return candidates[np.argmax(self.fitness[candidates])]

    def _row_crossover(self, p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
        """Row-wise crossover: random cut-point per row.

        For each row, a random column index is chosen. Columns left of the
        cut come from parent 1, columns right from parent 2 (reference style).
        """
        g1 = p1.reshape(self.inner, self.inner)
        g2 = p2.reshape(self.inner, self.inner)
        child = np.empty_like(g1)
        cuts = self.rng.integers(0, self.inner, size=self.inner)
        for r in range(self.inner):
            c = cuts[r]
            child[r, :c] = g1[r, :c]
            child[r, c:] = g2[r, c:]
        return child.flatten()

    def _col_crossover(self, p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
        """Column-wise crossover: random cut-point per column.

        For each column, a random row index is chosen. Rows above the cut
        come from parent 1, rows below from parent 2.
        """
        g1 = p1.reshape(self.inner, self.inner)
        g2 = p2.reshape(self.inner, self.inner)
        child = np.empty_like(g1)
        cuts = self.rng.integers(0, self.inner, size=self.inner)
        for c in range(self.inner):
            r = cuts[c]
            child[:r, c] = g1[:r, c]
            child[r:, c] = g2[r:, c]
        return child.flatten()

    def _uniform_crossover(self, p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
        """Uniform crossover: each bit from either parent with 50% probability."""
        mask = self.rng.random(self.n_bits) < 0.5
        child = np.where(mask, p1, p2)
        return child

    def _block_crossover(self, p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
        """2D block crossover: copy a rectangular block from p2 into p1.

        Captures spatial locality — contiguous pixel regions tend to form
        resonant structures together.
        """
        child = p1.copy()
        g1 = child.reshape(self.inner, self.inner)

        bmin, bmax = self.config.block_size_range
        bw = self.rng.integers(bmin, bmax + 1)
        bh = self.rng.integers(bmin, bmax + 1)
        x0 = self.rng.integers(0, self.inner - bw + 1)
        y0 = self.rng.integers(0, self.inner - bh + 1)

        g2 = p2.reshape(self.inner, self.inner)
        g1[y0:y0+bh, x0:x0+bw] = g2[y0:y0+bh, x0:x0+bw]
        return g1.flatten()

    def _mutate(self, genome: np.ndarray) -> np.ndarray:
        """Bit-flip mutation with adjacency bias.

        With 50% probability, mutations are biased to flip pixels adjacent
        to existing metal — this grows/shrinks structures rather than
        creating disconnected noise pixels.
        """
        genome = genome.copy()

        if self.rng.random() < 0.5:
            # Adjacency-biased mutation: flip pixels near existing metal
            grid_2d = genome.reshape(self.inner, self.inner)
            # Find pixels adjacent to metal (dilate by 1)
            from scipy.ndimage import binary_dilation
            dilated = binary_dilation(grid_2d, iterations=1)
            border = dilated & ~grid_2d.astype(bool)  # empty pixels next to metal
            border_flat = border.flatten()

            n_flips = max(1, int(self.mutation_rate * self.n_bits))
            # Mix: some grow (add metal next to existing), some shrink (remove edge metal)
            candidates_add = np.where(border_flat)[0]
            candidates_rm = np.where(genome == 1)[0]

            # 60% add, 40% remove
            n_add = int(n_flips * 0.6)
            n_rm = n_flips - n_add

            if len(candidates_add) > 0 and n_add > 0:
                idx = self.rng.choice(candidates_add, min(n_add, len(candidates_add)), replace=False)
                genome[idx] = 1
            if len(candidates_rm) > 0 and n_rm > 0:
                idx = self.rng.choice(candidates_rm, min(n_rm, len(candidates_rm)), replace=False)
                genome[idx] = 0
        else:
            # Standard random bit-flip
            flip_mask = self.rng.random(self.n_bits) < self.mutation_rate
            genome[flip_mask] = 1 - genome[flip_mask]

        # Enforce fill constraints
        fill = genome.mean()
        if fill < self.config.min_fill:
            n_needed = int(self.config.min_fill * self.n_bits) - genome.sum()
            zeros = np.where(genome == 0)[0]
            if len(zeros) > 0:
                to_flip = self.rng.choice(zeros, min(n_needed, len(zeros)), replace=False)
                genome[to_flip] = 1
        elif fill > self.config.max_fill:
            n_excess = genome.sum() - int(self.config.max_fill * self.n_bits)
            ones = np.where(genome == 1)[0]
            if len(ones) > 0:
                to_flip = self.rng.choice(ones, min(n_excess, len(ones)), replace=False)
                genome[to_flip] = 0

        return genome

    def _crossover_ports(self, p1_ports: np.ndarray, p2_ports: np.ndarray) -> np.ndarray:
        """Crossover port positions: each port from either parent with 50% probability."""
        mask = self.rng.random(4) < 0.5
        return np.where(mask, p1_ports, p2_ports)

    def _inherit_ports_gaussian(self, p1_ports: np.ndarray, p2_ports: np.ndarray) -> np.ndarray:
        """Inherit port positions from random parent + Gaussian jitter.

        Reference style: pick each port from a random parent, add N(0, σ²)
        jitter, round, clamp to valid range [2, inner-2].
        """
        sigma = self.config.port_jitter_sigma
        mask = self.rng.random(4) < 0.5
        ports = np.where(mask, p1_ports, p2_ports).astype(np.float64)
        ports += self.rng.normal(0, sigma, size=4)
        ports = np.round(ports).astype(np.int32)
        ports = np.clip(ports, 2, self.inner - 2)
        return ports

    def _mutate_ports(self, ports: np.ndarray) -> np.ndarray:
        """Mutate port positions: small random shift with probability = mutation_rate."""
        ports = ports.copy()
        for i in range(4):
            if self.rng.random() < self.mutation_rate:
                # Shift by +/- 1-3 positions
                delta = int(self.rng.integers(-3, 4))
                ports[i] = np.clip(ports[i] + delta, 1, self.inner)
        return ports

    def step(self) -> dict:
        """Run one generation. Returns stats dict."""
        if self.population is None:
            self._init_population()
            self.fitness = self._evaluate_population()
            self.generation = 0

        n_elite = max(1, int(self.config.pop_size * self.config.elite_frac))

        # Sort by fitness (descending)
        order = np.argsort(-self.fitness)
        elite_indices = order[:n_elite]

        # Build next generation
        new_pop = np.zeros_like(self.population)
        new_ports = np.zeros_like(self.port_positions)

        # Copy elites
        for i, idx in enumerate(elite_indices):
            new_pop[i] = self.population[idx].copy()
            new_ports[i] = self.port_positions[idx].copy()

        # Fill rest with crossover + mutation
        crossover_mode = self.config.crossover_mode
        use_gaussian_ports = self.config.port_jitter_sigma > 0

        for i in range(n_elite, self.config.pop_size):
            p1_idx = self._tournament_select()
            p2_idx = self._tournament_select()

            if self.rng.random() < self.config.crossover_rate:
                if crossover_mode == "row":
                    child = self._row_crossover(
                        self.population[p1_idx], self.population[p2_idx]
                    )
                elif crossover_mode == "col":
                    child = self._col_crossover(
                        self.population[p1_idx], self.population[p2_idx]
                    )
                elif crossover_mode == "rowcol":
                    if self.rng.random() < 0.5:
                        child = self._row_crossover(
                            self.population[p1_idx], self.population[p2_idx]
                        )
                    else:
                        child = self._col_crossover(
                            self.population[p1_idx], self.population[p2_idx]
                        )
                elif self.rng.random() < self.config.block_crossover_prob:
                    child = self._block_crossover(
                        self.population[p1_idx], self.population[p2_idx]
                    )
                else:
                    child = self._uniform_crossover(
                        self.population[p1_idx], self.population[p2_idx]
                    )

                if use_gaussian_ports:
                    child_ports = self._inherit_ports_gaussian(
                        self.port_positions[p1_idx], self.port_positions[p2_idx]
                    )
                else:
                    child_ports = self._crossover_ports(
                        self.port_positions[p1_idx], self.port_positions[p2_idx]
                    )
            else:
                child = self.population[p1_idx].copy()
                child_ports = self.port_positions[p1_idx].copy()

            child = self._mutate(child)
            if not use_gaussian_ports:
                child_ports = self._mutate_ports(child_ports)
            new_pop[i] = child
            new_ports[i] = child_ports

        self.population = new_pop
        self.port_positions = new_ports
        self.fitness = self._evaluate_population()
        self.generation += 1

        # Decay mutation rate
        if self.config.mutation_decay_mode == "linear":
            # Linear decay: rate = initial - gen/n_gen * delta, clamped ≥ min
            progress = self.generation / max(self.config.n_generations, 1)
            self.mutation_rate = max(
                self.config.min_mutation_rate,
                self.config.mutation_rate - progress * self.config.linear_decay_delta,
            )
        else:
            # Multiplicative decay (legacy)
            self.mutation_rate = max(
                self.config.min_mutation_rate,
                self.mutation_rate * self.config.mutation_decay,
            )

        best_idx = np.argmax(self.fitness)
        self.best_fitness_history.append(float(self.fitness[best_idx]))
        self.mean_fitness_history.append(float(self.fitness.mean()))

        return {
            "generation": self.generation,
            "best_fitness": float(self.fitness[best_idx]),
            "mean_fitness": float(self.fitness.mean()),
            "std_fitness": float(self.fitness.std()),
            "mutation_rate": self.mutation_rate,
            "best_fill": float(self.population[best_idx].mean()),
        }

    def run(self, verbose: bool = True, progress_callback=None) -> dict:
        """Run full GA optimization. Returns final results.

        progress_callback: optional callable(phase:str, gen:int, total:int,
                                              stats:dict|None) invoked per
        generation and at local-search/finalization milestones. Exceptions
        raised by the callback are swallowed so GA progress isn't affected.
        """
        t0 = time.time()

        def _notify(phase, gen, total, stats=None):
            if progress_callback is None:
                return
            try:
                progress_callback(phase, gen, total, stats)
            except Exception:
                pass

        if self.population is None:
            self._init_population()
            self.fitness = self._evaluate_population()

            best_idx = np.argmax(self.fitness)
            self.best_fitness_history.append(float(self.fitness[best_idx]))
            self.mean_fitness_history.append(float(self.fitness.mean()))

            if verbose:
                print(f"Gen   0 | best={self.fitness[best_idx]:.4f} "
                      f"mean={self.fitness.mean():.4f} "
                      f"fill={self.population[best_idx].mean():.2f}")

            _notify("init", 0, self.config.n_generations, {
                "best_fitness": float(self.fitness[best_idx]),
                "mean_fitness": float(self.fitness.mean()),
            })

        for gen in range(1, self.config.n_generations + 1):
            stats = self.step()

            if verbose and (gen % 20 == 0 or gen == 1 or gen == self.config.n_generations):
                print(f"Gen {gen:3d} | best={stats['best_fitness']:.4f} "
                      f"mean={stats['mean_fitness']:.4f} "
                      f"mut={stats['mutation_rate']:.4f} "
                      f"fill={stats['best_fill']:.2f}")

            _notify("gen", gen, self.config.n_generations, stats)

        # Local search: multi-pass greedy single-pixel hill-climb on top designs
        if self.config.local_search:
            if verbose:
                print("\nLocal search (multi-pass single-pixel hill-climb)...")
            _notify("local_search", 0, self.config.local_search_k, None)
            t_local = time.time()
            n_local = min(self.config.local_search_k, self.config.pop_size)
            local_order = np.argsort(-self.fitness)[:n_local]
            total_improved = 0
            max_passes = 3

            def _scored(grid_batch, ports_row):
                """Raw-fitness minus DC-path + stranded-metal penalties, so
                hill-climb optimises the same landscape that selection sees
                and cannot undo connectivity / clean-trace gains."""
                sp = self.evaluator.predict_sparams(grid_batch)
                base = np.array([
                    self.objective.evaluate(sp[i])["fitness"]
                    for i in range(len(grid_batch))
                ])
                ports_batch = np.tile(ports_row, (len(grid_batch), 1))
                base -= self._dc_path_penalty(grid_batch, ports_batch)
                base -= self._stranded_metal_penalty(grid_batch, ports_batch)
                return base

            for li, idx in enumerate(local_order):
                genome = self.population[idx].copy()
                cur_score = float(self.fitness[idx])

                ports = self.port_positions[idx]
                for pass_num in range(max_passes):
                    variants = np.tile(genome, (self.n_bits, 1))
                    np.fill_diagonal(variants, 1 - np.diag(variants))
                    grids = np.array([self._make_grid(v, ports) for v in variants])
                    scores = _scored(grids, ports)

                    benefit = scores - cur_score
                    improving = np.where(benefit > 0)[0]
                    pass_improved = 0
                    if len(improving) > 0:
                        for px in improving[np.argsort(-benefit[improving])]:
                            genome[px] = 1 - genome[px]
                            grid = self._make_grid(genome, ports)
                            new_score = float(_scored(grid[np.newaxis], ports)[0])
                            if new_score > cur_score:
                                cur_score = new_score
                                pass_improved += 1
                            else:
                                genome[px] = 1 - genome[px]

                    total_improved += pass_improved
                    if pass_improved == 0:
                        break

                self.population[idx] = genome
                self.fitness[idx] = cur_score

            dt_local = time.time() - t_local
            if verbose:
                print(f"  Local search: {total_improved} improvements on {n_local} "
                      f"designs ({max_passes} max passes) in {dt_local:.1f}s")

        dt = time.time() - t0

        # Get top-K results
        order = np.argsort(-self.fitness)
        top_k = 10
        results = []
        for rank, idx in enumerate(order[:top_k]):
            grid = self._make_grid(self.population[idx], self.port_positions[idx])
            sparams = self.evaluator.predict_sparams(grid[np.newaxis])[0]
            score = self.objective.evaluate(sparams)
            results.append({
                "rank": rank,
                "fitness": float(self.fitness[idx]),
                "fill": float(self.population[idx].mean()),
                "grid": grid,
                "sparams": sparams,
                "goals": score["goals"],
            })

        if verbose:
            print(f"\nOptimization complete in {dt:.1f}s")
            print(f"Best fitness: {results[0]['fitness']:.4f}")
            for g in results[0]["goals"]:
                print(f"  {g['goal']}: achieved={g['achieved_db']:.1f}dB "
                      f"worst={g.get('worst_db', float('nan')):.1f}dB")

        return {
            "config": self.config,
            "objective": self.objective.name,
            "n_generations": self.config.n_generations,
            "time_s": dt,
            "best_fitness_history": self.best_fitness_history,
            "mean_fitness_history": self.mean_fitness_history,
            "top_k": results,
        }
