"""Unified simulation interface for running FDTD on pixel grids.

All code that needs to simulate a grid and get S-parameters back should
use `simulate_grid()` with a `SimConfig`. This avoids duplicating FDTD
parameter passing across optimizer, evaluate, and dataset modules.

Three preset configs cover common use cases:
  - DATASET_CONFIG: standard fidelity for training data generation
  - VALIDATION_CONFIG: medium fidelity for GA candidate validation
  - HIFI_CONFIG: high fidelity for final design verification
"""

from __future__ import annotations

import shutil
import time
import traceback
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class SimConfig:
    """FDTD simulation parameters."""
    nr_ts: int = 500_000
    end_criteria_db: float = -35.0
    n_freq: int = 30
    f_start: float = 1e9
    f_stop: float = 30e9
    mesh_div: float = 20.0
    use_pml: bool = True
    xy_pad_um: float = 200.0
    num_ports: int = 4
    keep_sim_dir: bool = False


# ── Presets ──────────────────────────────────────────────────────────────

DATASET_CONFIG = SimConfig(
    nr_ts=500_000,
    end_criteria_db=-35.0,
    n_freq=30,
    mesh_div=20.0,
)

VALIDATION_CONFIG = SimConfig(
    nr_ts=1_000_000,
    end_criteria_db=-50.0,
    n_freq=30,
    mesh_div=30.0,
)

HIFI_CONFIG = SimConfig(
    nr_ts=5_000_000,
    end_criteria_db=-60.0,
    n_freq=200,
    f_start=0.1e9,
    f_stop=40e9,
    mesh_div=40.0,
)


def simulate_grid(
    grid: np.ndarray,
    work_dir: Path,
    config: SimConfig | None = None,
) -> dict:
    """Run FDTD simulation on a pixel grid and return S-parameters.

    Parameters
    ----------
    grid : (outer, outer) binary pixel grid
    work_dir : directory for simulation files (created if needed)
    config : simulation parameters (defaults to DATASET_CONFIG)

    Returns
    -------
    dict with keys:
        success : bool
        sparams : (n_freq, 4, 4) complex array (if success)
        validation_info : dict from openems_pixel (if success)
        time_s : float
        error : str (if not success)
    """
    from invdesign.openems_pixel import run_openems_pixel_grid
    from surrogate.data import load_design

    config = config or DATASET_CONFIG
    work_dir = Path(work_dir).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)

    np.save(work_dir / "pixel_grid.npy", grid)
    out_prefix = work_dir / "pixel_grid"

    t0 = time.time()
    try:
        val_info = run_openems_pixel_grid(
            out_prefix=out_prefix,
            grid=grid,
            nr_ts=config.nr_ts,
            end_criteria_db=config.end_criteria_db,
            n_freq=config.n_freq,
            f_start=config.f_start,
            f_stop=config.f_stop,
            mesh_div=config.mesh_div,
            use_pml=config.use_pml,
            xy_pad_um=config.xy_pad_um,
            num_ports=config.num_ports,
        )
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc(),
            "time_s": time.time() - t0,
        }

    sim_time = time.time() - t0

    # Load S-parameters from touchstone file
    loaded = load_design(work_dir)
    if loaded is None:
        return {
            "success": False,
            "error": f"Failed to parse S-parameters from {work_dir}",
            "time_s": sim_time,
        }

    _, sparams = loaded

    # Clean up simulation directory
    sim_dir = work_dir / "pixel_grid_pixel_sim"
    if sim_dir.exists() and not config.keep_sim_dir:
        shutil.rmtree(sim_dir)

    return {
        "success": True,
        "sparams": sparams,
        "validation_info": val_info,
        "time_s": sim_time,
    }
