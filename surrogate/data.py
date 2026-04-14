"""Dataset loader for pixel-grid → S-parameter surrogate training.

Key design decisions:

1. **Reciprocity encoding**: S-params stored as upper triangle (10 unique
   complex values per freq), enforcing S_ij = S_ji. This halves the output
   dimensionality and encodes the passive-network physics.

2. **D4 symmetry augmentation**: The pixel grid has 8-fold symmetry
   (4 rotations × 2 reflections). Each geometric transform permutes the
   port ordering (N=0, S=1, E=2, W=3) and thus permutes the S-matrix.
   This gives 8× data augmentation for free.

3. **Normalization**: S-params span huge dynamic range (|S11|≈1, |S21|≈0.001).
   We normalize per-element using dataset statistics (mean/std) computed at
   init time. This lets the network use MSE loss without bias toward S11.

4. **Fill-balanced sampling**: When combining datasets with different fill
   distributions (e.g. V1 fixed 50% + V2 variable 10-90%), we provide a
   WeightedRandomSampler that equalizes representation across fill bins.
   Without this, the ~50% fill bin dominates training.

5. **Stratified evaluation**: fill_bins attribute enables per-bin evaluation
   to verify the model learns across the full fill range, not just ~50%.

Upper triangle of 4×4 matrix = 10 entries:
  idx: (0,0) (0,1) (0,2) (0,3) (1,1) (1,2) (1,3) (2,2) (2,3) (3,3)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, WeightedRandomSampler


# Upper-triangle indices for 4×4 matrix
_UTRI_ROWS, _UTRI_COLS = np.triu_indices(4)  # 10 pairs

# Fill-factor bin edges for balanced sampling and stratified eval.
# Bins: sparse (<0.35), mid (0.35–0.65), dense (>0.65)
FILL_BIN_EDGES = [0.0, 0.35, 0.65, 1.0]
FILL_BIN_NAMES = ["sparse (<35%)", "mid (35-65%)", "dense (>65%)"]

# ---------------------------------------------------------------------------
# D4 symmetry: 8 transforms of the pixel grid and corresponding port permutations
# Ports: 0=North, 1=South, 2=East, 3=West
# ---------------------------------------------------------------------------

# (grid_transform_fn, port_permutation)
# Grid transforms operate on (27, 27) numpy arrays.
_D4_TRANSFORMS: list[tuple[str, list[int]]] = [
    # identity
    ("identity",    [0, 1, 2, 3]),
    # 90° CW rotation (np.rot90 k=-1): N→E, E→S, S→W, W→N
    ("rot90_cw",    [3, 2, 0, 1]),
    # 180° rotation: N→S, S→N, E→W, W→E
    ("rot180",      [1, 0, 3, 2]),
    # 270° CW rotation (= 90° CCW): N→W, W→S, S→E, E→N
    ("rot270_cw",   [2, 3, 1, 0]),
    # Horizontal flip (left-right): E↔W
    ("flip_lr",     [0, 1, 3, 2]),
    # Vertical flip (up-down): N↔S
    ("flip_ud",     [1, 0, 2, 3]),
    # Transpose (flip along main diagonal): N↔W, S↔E → swap (0,3) and (1,2)
    ("transpose",   [3, 2, 0, 1]),  # same perm as rot90 but different grid op
    # Anti-transpose (flip along anti-diagonal)
    ("anti_transpose", [2, 3, 1, 0]),
]


def _apply_grid_transform(grid: np.ndarray, name: str) -> np.ndarray:
    """Apply a D4 spatial transform to a 2D grid."""
    if name == "identity":
        return grid
    elif name == "rot90_cw":
        return np.rot90(grid, k=-1)
    elif name == "rot180":
        return np.rot90(grid, k=2)
    elif name == "rot270_cw":
        return np.rot90(grid, k=1)
    elif name == "flip_lr":
        return np.fliplr(grid)
    elif name == "flip_ud":
        return np.flipud(grid)
    elif name == "transpose":
        return grid.T
    elif name == "anti_transpose":
        return np.rot90(np.fliplr(grid), k=1)
    raise ValueError(f"Unknown transform: {name}")


def _permute_sparams(sparams: np.ndarray, perm: list[int]) -> np.ndarray:
    """Permute rows and columns of S-matrix according to port reordering.

    sparams: (n_freq, 4, 4) complex
    perm: new port order, e.g. [1, 0, 3, 2] for 180° rotation
    """
    return sparams[:, perm, :][:, :, perm]


def _inner_fill(grid: np.ndarray) -> float:
    """Fill factor of the inner region (excluding 1-pixel port border)."""
    return float(grid[1:-1, 1:-1].mean())


def _fill_bin(fill: float) -> int:
    """Map fill factor to bin index: 0=sparse, 1=mid, 2=dense."""
    for i in range(len(FILL_BIN_EDGES) - 1):
        if fill < FILL_BIN_EDGES[i + 1]:
            return i
    return len(FILL_BIN_EDGES) - 2


def load_design(design_dir: Path) -> tuple[np.ndarray, np.ndarray] | None:
    """Load a single design. Returns (grid, sparams) or None if missing."""
    npy = design_dir / "pixel_grid.npy"
    s4p = design_dir / "pixel_grid.s4p"
    if not npy.exists() or not s4p.exists():
        return None

    grid = np.load(npy)  # (27, 27), int8

    # Parse touchstone inline
    freqs = []
    data = []
    current = []
    n_vals = 4 * 4 * 2  # 32 values per freq point
    with s4p.open() as f:
        for line in f:
            line = line.strip()
            if not line or line[0] in "#!":
                continue
            parts = line.split()
            vals = [float(v) for v in parts]
            if not current:
                freqs.append(vals[0])
                current.extend(vals[1:])
            else:
                current.extend(vals)
            if len(current) >= n_vals:
                data.append(current[:n_vals])
                current = []

    n_freq = len(freqs)
    sparams = np.zeros((n_freq, 4, 4), dtype=np.complex128)
    for idx, row in enumerate(data):
        k = 0
        for i in range(4):
            for j in range(4):
                sparams[idx, i, j] = row[k] + 1j * row[k + 1]
                k += 2

    return grid, sparams


def sparams_to_upper_tri(sparams: np.ndarray) -> np.ndarray:
    """Extract upper triangle → (n_freq, 10, 2) float32 [real, imag]."""
    sparams_sym = 0.5 * (sparams + sparams.transpose(0, 2, 1))
    utri = sparams_sym[:, _UTRI_ROWS, _UTRI_COLS]  # (n_freq, 10)
    return np.stack([utri.real, utri.imag], axis=-1).astype(np.float32)


def upper_tri_to_sparams(utri: np.ndarray) -> np.ndarray:
    """Reconstruct full 4×4 from upper triangle (n_freq, 10, 2)."""
    n_freq = utri.shape[0]
    sparams = np.zeros((n_freq, 4, 4), dtype=np.complex128)
    cpx = utri[..., 0] + 1j * utri[..., 1]
    sparams[:, _UTRI_ROWS, _UTRI_COLS] = cpx
    sparams[:, _UTRI_COLS, _UTRI_ROWS] = cpx
    return sparams


class PixelGridDataset(Dataset):
    """PyTorch dataset with D4 symmetry augmentation and per-element normalization.

    Each raw design produces 8 augmented samples (4 rotations × 2 reflections).
    Fill factor is tracked per-sample for balanced sampling and stratified eval.
    Coupling weight is computed per-sample: higher for designs with meaningful
    port-to-port coupling (GA-relevant), lower for disconnected/weak designs.

    Input:  (1, 27, 27) float32 — binary pixel grid
    Target: (30, 10, 2) float32 — normalized upper-tri S-params
    """

    def __init__(
        self,
        dataset_dir: str | Path | list[str | Path],
        max_designs: int | None = None,
        augment: bool = True,
        normalize: bool = True,
        coupling_weight_high: float = 5.0,
        coupling_threshold_db: float = -30.0,
        fill_min: float = 0.0,
        fill_max: float = 1.0,
        disconnect_aug_prob: float = 0.0,
        disconnect_aug_db: float = -80.0,
    ):
        self.augment = augment
        self.normalize = normalize
        self.fill_min = float(fill_min)
        self.fill_max = float(fill_max)
        self.disconnect_aug_prob = float(disconnect_aug_prob)
        self.disconnect_aug_db = float(disconnect_aug_db)
        self._disc_rng = np.random.default_rng()
        self.grids: list[np.ndarray] = []
        self.targets: list[np.ndarray] = []
        self.fill_factors: list[float] = []
        self.fill_bins: list[int] = []
        self.coupling_weights: list[float] = []
        n_fill_rejected = 0

        # Support single or multiple dataset directories
        if isinstance(dataset_dir, (str, Path)):
            dirs = [Path(dataset_dir)]
        else:
            dirs = [Path(d) for d in dataset_dir]

        design_dirs = []
        for d in dirs:
            design_dirs.extend(sorted(d.glob("design_*")))
        # Shuffle when combining multiple dirs so max_designs draws evenly
        if len(dirs) > 1:
            rng = np.random.default_rng(0)
            rng.shuffle(design_dirs)
        if max_designs is not None:
            design_dirs = design_dirs[:max_designs]

        n_loaded = 0
        n_well_coupled = 0
        for d in design_dirs:
            result = load_design(d)
            if result is None:
                continue
            grid, sparams = result
            fill = _inner_fill(grid)
            if fill < self.fill_min or fill > self.fill_max:
                n_fill_rejected += 1
                continue
            fbin = _fill_bin(fill)

            # Coupling weight: check max off-diagonal |S_ij| across all port pairs
            # If any port pair has coupling above threshold, it's GA-relevant
            off_diag_mag = np.abs(sparams[:, _UTRI_ROWS[_UTRI_ROWS != _UTRI_COLS],
                                           _UTRI_COLS[_UTRI_ROWS != _UTRI_COLS]])
            max_coupling_db = 20 * np.log10(off_diag_mag.max() + 1e-12)
            is_well_coupled = max_coupling_db > coupling_threshold_db
            cw = coupling_weight_high if is_well_coupled else 1.0
            if is_well_coupled:
                n_well_coupled += 1

            if augment:
                transforms = _D4_TRANSFORMS
            else:
                transforms = [_D4_TRANSFORMS[0]]  # identity only

            for tfm_name, port_perm in transforms:
                g = _apply_grid_transform(grid, tfm_name).copy()
                sp = _permute_sparams(sparams, port_perm)
                self.grids.append(g.astype(np.float32))
                self.targets.append(sparams_to_upper_tri(sp))
                # Fill factor and coupling weight are invariant under D4 transforms
                self.fill_factors.append(fill)
                self.fill_bins.append(fbin)
                self.coupling_weights.append(cw)

            n_loaded += 1

        n_aug = len(self.grids)
        self.fill_factors = np.array(self.fill_factors, dtype=np.float32)
        self.fill_bins = np.array(self.fill_bins, dtype=np.int64)
        self.coupling_weights = np.array(self.coupling_weights, dtype=np.float32)

        # Print fill distribution
        bin_counts = np.bincount(self.fill_bins, minlength=len(FILL_BIN_NAMES))
        fill_str = ", ".join(f"{FILL_BIN_NAMES[i]}={bin_counts[i]}" for i in range(len(FILL_BIN_NAMES)))
        print(f"Loaded {n_loaded} designs → {n_aug} samples "
              f"({'8× augmented' if augment else 'no augmentation'})")
        if self.fill_min > 0.0 or self.fill_max < 1.0:
            print(f"  Fill clamp [{self.fill_min:.2f}, {self.fill_max:.2f}]: "
                  f"rejected {n_fill_rejected} designs")
        print(f"  Fill distribution: {fill_str}")
        print(f"  Well-coupled (S_ij > {coupling_threshold_db:.0f}dB): "
              f"{n_well_coupled}/{n_loaded} ({100*n_well_coupled/max(n_loaded,1):.0f}%) "
              f"— weighted {coupling_weight_high}×")

        # Compute per-element normalization statistics
        if normalize and self.targets:
            all_targets = np.stack(self.targets)  # (N, 30, 10, 2)
            self.target_mean = all_targets.mean(axis=0)  # (30, 10, 2)
            self.target_std = all_targets.std(axis=0) + 1e-8  # avoid /0
            # Apply normalization
            for i in range(len(self.targets)):
                self.targets[i] = (self.targets[i] - self.target_mean) / self.target_std
        else:
            self.target_mean = np.zeros((30, 10, 2), dtype=np.float32)
            self.target_std = np.ones((30, 10, 2), dtype=np.float32)

    def __len__(self) -> int:
        return len(self.grids)

    def _make_disconnected_sample(self) -> tuple[np.ndarray, np.ndarray]:
        """Synthetic disconnected-grid sample: speckle + near-zero coupling.

        Cures the surrogate's 'hallucinate ~-38 dB on empty grids' failure
        mode without new OpenEMS runs. Target is S_ii=1 (total reflection),
        S_ij = 10^(disconnect_aug_db/20) with zero phase.
        """
        outer = self.grids[0].shape[0]
        rng = self._disc_rng
        grid = np.zeros((outer, outer), dtype=np.float32)
        n = int(rng.integers(0, 20))
        for _ in range(n):
            r = int(rng.integers(2, outer - 2))
            c = int(rng.integers(2, outer - 2))
            grid[r, c] = 1.0
        n_freq = self.targets[0].shape[0]
        mag = 10 ** (self.disconnect_aug_db / 20.0)
        sparams = np.zeros((n_freq, 4, 4), dtype=np.complex128)
        for i in range(4):
            sparams[:, i, i] = 1.0 + 0j
            for j in range(4):
                if i != j:
                    sparams[:, i, j] = mag + 0j
        target = sparams_to_upper_tri(sparams)  # (n_freq, 10, 2)
        if self.normalize:
            target = (target - self.target_mean) / self.target_std
        return grid, target.astype(np.float32)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.disconnect_aug_prob > 0.0 and self._disc_rng.random() < self.disconnect_aug_prob:
            g, t = self._make_disconnected_sample()
            grid = torch.from_numpy(g).unsqueeze(0)
            target = torch.from_numpy(t)
            # Weight at coupling_weight=1.0 (not 'well coupled' - low S_ij)
            return grid, target, torch.tensor(1.0)
        grid = torch.from_numpy(self.grids[idx]).unsqueeze(0)  # (1, 27, 27)
        target = torch.from_numpy(self.targets[idx])  # (30, 10, 2)
        weight = torch.tensor(self.coupling_weights[idx])  # scalar
        return grid, target, weight

    def denormalize(self, pred: torch.Tensor) -> torch.Tensor:
        """Convert normalized predictions back to physical S-param scale."""
        mean = torch.from_numpy(self.target_mean).to(pred.device)
        std = torch.from_numpy(self.target_std).to(pred.device)
        return pred * std + mean

    def get_balanced_sampler(self) -> WeightedRandomSampler:
        """Return a sampler that equalizes fill-bin representation per epoch.

        Without this, combining V1 (all ~50% fill) + V2 (10-90% fill) would
        over-represent the mid-fill bin by ~3:1 vs sparse/dense.
        """
        bin_counts = np.bincount(self.fill_bins, minlength=len(FILL_BIN_NAMES))
        # Weight = 1 / bin_count → each bin contributes equally per epoch
        bin_weights = 1.0 / np.maximum(bin_counts, 1).astype(np.float64)
        sample_weights = bin_weights[self.fill_bins]
        return WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(self),
            replacement=True,
        )

    def get_fill_bin_indices(self) -> dict[str, np.ndarray]:
        """Return sample indices grouped by fill bin, for stratified evaluation."""
        result = {}
        for i, name in enumerate(FILL_BIN_NAMES):
            result[name] = np.where(self.fill_bins == i)[0]
        return result
