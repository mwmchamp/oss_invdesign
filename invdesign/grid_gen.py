"""Random binary pixel grids with edge ports.

Includes both purely random grids (original) and path-seeded grids
that guarantee metal connectivity between specified port pairs.

Port positions are ALWAYS randomized — one per edge, placed at a random
inner position (columns/rows 1 to inner_size).
"""

from __future__ import annotations

import numpy as np


def _pick_port_positions(
    rng: np.random.Generator,
    inner_size: int,
) -> dict[int, int]:
    """Pick one random port position per edge.

    Returns dict mapping port index (0=N, 1=S, 2=E, 3=W) to the
    position along that edge (value in [1, inner_size], i.e. valid
    inner-grid coordinates).
    """
    return {
        0: int(rng.integers(1, inner_size + 1)),  # N: column
        1: int(rng.integers(1, inner_size + 1)),  # S: column
        2: int(rng.integers(1, inner_size + 1)),  # E: row
        3: int(rng.integers(1, inner_size + 1)),  # W: row
    }


def place_ports(
    grid: np.ndarray,
    port_positions: dict[int, int],
    ports: set[int] | None = None,
) -> None:
    """Place port border pixels + inner adjacency on the grid (in-place).

    Parameters
    ----------
    grid : (outer, outer) array
    port_positions : dict mapping port index to position along edge
    ports : which ports to place (default: all 4)
    """
    outer = grid.shape[0]
    if ports is None:
        ports = {0, 1, 2, 3}
    for pi in ports:
        pos = port_positions[pi]
        if pi == 0:    # N: top row
            grid[0, pos] = 1
            grid[1, pos] = 1
        elif pi == 1:  # S: bottom row
            grid[outer - 1, pos] = 1
            grid[outer - 2, pos] = 1
        elif pi == 2:  # E: right col
            grid[pos, outer - 1] = 1
            grid[pos, outer - 2] = 1
        elif pi == 3:  # W: left col
            grid[pos, 0] = 1
            grid[pos, 1] = 1


def generate_pixel_grid(
    seed: int | None,
    inner_size: int,
    ports_per_edge: int = 1,
    *,
    fill_range: tuple[float, float] = (0.1, 0.9),
    fill_mean: float = 0.5,
    fill_std: float = 0.15,
) -> np.ndarray:
    """Return ``(inner+2)x(inner+2)`` binary array; border rows/cols hold ports.

    Port positions are randomized — one per edge at a random inner position.

    Parameters
    ----------
    fill_range : tuple[float, float]
        Hard clamp on fill fraction (samples outside are clipped).
    fill_mean : float
        Mean of the normal distribution for fill factor sampling.
    fill_std : float
        Standard deviation of the normal distribution for fill factor sampling.
    """
    rng = np.random.default_rng(seed)
    outer = inner_size + 2
    grid = np.zeros((outer, outer), dtype=np.int8)

    # Sample fill factor from N(mean, std^2), clamp to valid range
    fill = rng.normal(fill_mean, fill_std)
    fill = float(np.clip(fill, fill_range[0], fill_range[1]))
    grid[1:-1, 1:-1] = (rng.random(size=(inner_size, inner_size)) < fill).astype(np.int8)

    # Random port placement with inner adjacency
    port_positions = _pick_port_positions(rng, inner_size)
    place_ports(grid, port_positions)

    return grid


def _random_walk_path(
    rng: np.random.Generator,
    grid: np.ndarray,
    start: tuple[int, int],
    end: tuple[int, int],
    inner_size: int,
    width: int = 1,
) -> None:
    """Draw a random-walk metal path between two points on the inner grid.

    The path uses Manhattan moves (up/down/left/right) with a bias toward
    the target to avoid wandering too far. Path width can be > 1 for wider
    waveguide-like structures.
    """
    r, c = start
    er, ec = end
    max_steps = inner_size * 4  # prevent infinite loops

    for _ in range(max_steps):
        # Mark current position (and neighbors for wider paths)
        for dr in range(-(width // 2), width // 2 + 1):
            for dc in range(-(width // 2), width // 2 + 1):
                nr, nc = r + dr, c + dc
                if 1 <= nr < inner_size + 1 and 1 <= nc < inner_size + 1:
                    grid[nr, nc] = 1

        if r == er and c == ec:
            break

        # Biased random walk: 70% toward target, 30% random
        moves = []
        if r > er:
            moves.append((-1, 0))
        elif r < er:
            moves.append((1, 0))
        if c > ec:
            moves.append((0, -1))
        elif c < ec:
            moves.append((0, 1))

        if rng.random() < 0.3 or not moves:
            # Random orthogonal move
            all_moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]
            dr, dc = all_moves[rng.integers(4)]
        else:
            dr, dc = moves[rng.integers(len(moves))]

        nr, nc = r + dr, c + dc
        if 1 <= nr < inner_size + 1 and 1 <= nc < inner_size + 1:
            r, c = nr, nc


def generate_connected_grid(
    seed: int | None,
    inner_size: int,
    port_pairs: list[tuple[int, int]] | None = None,
    *,
    fill_range: tuple[float, float] = (0.1, 0.9),
    fill_mean: float = 0.5,
    fill_std: float = 0.15,
    path_width: int = 1,
) -> np.ndarray:
    """Generate a pixel grid with guaranteed metal paths between port pairs.

    Draws random-walk metal paths between the specified port pairs, then
    fills additional random pixels until the total inner fill fraction
    matches a target drawn from N(fill_mean, fill_std^2).

    The target fill is drawn from the same normal distribution as the
    purely-random grids (so datasets are comparable), but the lower bound
    is truncated at the path fill fraction — if the paths already occupy
    more metal than the sampled target, no pixels are removed.

    Parameters
    ----------
    port_pairs : list of (port_i, port_j) tuples
        Port indices: 0=N, 1=S, 2=E, 3=W. Default: [(0, 1)].
    fill_range : tuple[float, float]
        Hard clamp on sampled fill fraction (before path-floor truncation).
    fill_mean, fill_std : float
        Parameters of the normal distribution for target fill sampling.
    path_width : int
        Width of the waveguide path in pixels.
    """
    rng = np.random.default_rng(seed)
    outer = inner_size + 2
    grid = np.zeros((outer, outer), dtype=np.int8)

    if port_pairs is None:
        port_pairs = [(0, 1)]

    # Random port positions
    port_pos = _pick_port_positions(rng, inner_size)

    # Map port index to (row, col) on full grid for path endpoints
    port_coords = {
        0: (1, port_pos[0]),               # N: inner adjacency row
        1: (outer - 2, port_pos[1]),       # S: inner adjacency row
        2: (port_pos[2], outer - 2),       # E: inner adjacency col
        3: (port_pos[3], 1),               # W: inner adjacency col
    }

    # Place all 4 port border + adjacency pixels (simulation expects all ports)
    place_ports(grid, port_pos)

    # Draw paths between each port pair
    for pi, pj in port_pairs:
        start = port_coords[pi]
        end = port_coords[pj]
        pw = rng.integers(path_width, path_width + 2)
        _random_walk_path(rng, grid, start, end, inner_size, width=pw)

    # Sample target fill from the same normal distribution as random grids,
    # but floor it at the current path fill so we never remove path pixels.
    inner = grid[1:-1, 1:-1]
    n_inner = inner.size
    current_fill = float(inner.sum()) / n_inner

    target_fill = float(rng.normal(fill_mean, fill_std))
    target_fill = float(np.clip(target_fill, fill_range[0], fill_range[1]))
    target_fill = max(target_fill, current_fill)  # floor at path fill

    # Randomly flip empty pixels to metal until we reach the target fill.
    n_target = int(round(target_fill * n_inner))
    n_to_add = n_target - int(inner.sum())
    if n_to_add > 0:
        empty_idx = np.argwhere(inner == 0)
        if len(empty_idx) > 0:
            chosen = rng.choice(len(empty_idx), size=min(n_to_add, len(empty_idx)), replace=False)
            for idx in np.atleast_1d(chosen):
                r, c = empty_idx[int(idx)]
                grid[r + 1, c + 1] = 1

    return grid
