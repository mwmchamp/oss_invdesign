"""Batch generation of pixel-grid design folders (GDS + optional EM).

Each design folder contains:
  - pixel_grid.npy: binary (outer×outer) grid, dtype int8
  - pixel_grid.gds: GDSFactory layout
  - pixel_grid_pin_map.json: port name mapping
  - pixel_grid.s4p: Touchstone 4-port S-parameters (if EM backend = pixel)
  - pixel_grid.freqs.npy: frequency array (Hz)
  - meta.json: design metadata

The simulation directory (pixel_grid_pixel_sim/) is deleted after successful
S-parameter extraction to save disk space (~50 MB per design).
"""

from __future__ import annotations

import csv
import json
import shutil
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Literal

import numpy as np

from invdesign.grid_gen import generate_pixel_grid, generate_connected_grid
from invdesign.openems_pixel import run_openems_pixel_grid
from invdesign.openems_runner import add_paths, run_openems_waveguide

# Lazy imports for optional GDS/viz dependencies (gdsfactory needs pydantic v1)
def _import_layout_gf():
    from invdesign.layout_gf import write_layout_bundle
    return write_layout_bundle

def _import_layout_klayout():
    from invdesign.layout_klayout import write_klayout_gds
    return write_klayout_gds

def _import_viz():
    from invdesign.viz import render_gds_png
    return render_gds_png

EmBackend = Literal["none", "waveguide", "pixel"]


@dataclass
class DatasetConfig:
    output_dir: Path
    num_designs: int
    seed_start: int
    inner_size: int
    ports_per_edge: int
    em_backend: EmBackend
    install_prefix: Path
    nr_ts: int
    mesh_div: float
    write_klayout_copy: bool
    skip_png: bool
    num_ports: int = 4
    f_start: float = 1e9
    f_stop: float = 30e9
    n_freq: int = 30
    use_pml: bool = True
    end_criteria_db: float = -40.0
    xy_pad_um: float = 150.0
    keep_sim_dir: bool = False
    design_id_offset: int = 0
    connected_grids: bool = False


def generate_one(
    design_index: int,
    cfg: DatasetConfig,
) -> dict:
    """Create ``design_XXXXX`` folder; return manifest row dict."""
    global_index = cfg.design_id_offset + design_index
    seed = cfg.seed_start + design_index
    t0 = time.perf_counter()
    row: dict = {
        "design_id": f"design_{global_index:05d}",
        "seed": seed,
        "status": "ok",
        "error": "",
        "seconds": 0.0,
    }
    workdir = (cfg.output_dir / row["design_id"]).resolve()
    workdir.mkdir(parents=True, exist_ok=True)

    try:
        if cfg.connected_grids:
            # Random port pair selection for diverse coupling patterns
            import numpy as _np
            rng = _np.random.default_rng(seed)
            all_pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
            n_pairs = rng.integers(1, 3)  # 1-2 port pairs per design
            chosen = rng.choice(len(all_pairs), n_pairs, replace=False)
            port_pairs = [all_pairs[int(i)] for i in np.atleast_1d(chosen)]
            pw = int(rng.integers(1, 3))
            grid = generate_connected_grid(
                seed, cfg.inner_size,
                port_pairs=port_pairs,
                path_width=pw,
            )
        else:
            grid = generate_pixel_grid(
                seed,
                cfg.inner_size,
                cfg.ports_per_edge,
            )

        # GDS layout — optional, may fail if gdsfactory/pydantic incompatible
        gds_path = None
        npy_path = workdir / "pixel_grid.npy"
        try:
            write_layout_bundle = _import_layout_gf()
            gds_path, pin_json, npy_path = write_layout_bundle(
                grid, workdir, stem="pixel_grid", cell_name=f"PIXEL_GRID_{global_index:05d}",
            )
        except Exception:
            # Save grid directly when GDS generation unavailable
            np.save(npy_path, grid)

        if cfg.write_klayout_copy and gds_path is not None:
            write_klayout_gds = _import_layout_klayout()
            write_klayout_gds(grid, workdir, stem="pixel_grid_klayout")

        if not cfg.skip_png and gds_path is not None:
            render_gds_png = _import_viz()
            render_gds_png(gds_path, workdir / "pixel_grid.png")

        out_prefix = workdir / "pixel_grid"
        if cfg.em_backend == "waveguide":
            add_paths(cfg.install_prefix)
            run_openems_waveguide(out_prefix, nr_ts=cfg.nr_ts, mesh_div=cfg.mesh_div)
        elif cfg.em_backend == "pixel":
            add_paths(cfg.install_prefix)
            validation = run_openems_pixel_grid(
                out_prefix,
                grid,
                nr_ts=cfg.nr_ts,
                mesh_div=cfg.mesh_div,
                num_ports=cfg.num_ports,
                f_start=cfg.f_start,
                f_stop=cfg.f_stop,
                n_freq=cfg.n_freq,
                use_pml=cfg.use_pml,
                end_criteria_db=cfg.end_criteria_db,
                xy_pad_um=cfg.xy_pad_um,
            )
            if validation is not None:
                row["validation"] = validation
                if not validation.get("is_passive", False):
                    row["status"] = "warn_not_passive"
                if not validation.get("is_reciprocal", False):
                    row["status"] = row.get("status", "ok") if row["status"] != "ok" else "warn_not_reciprocal"

            # Clean up simulation directory to save disk space
            sim_dir = workdir / "pixel_grid_pixel_sim"
            if sim_dir.exists() and not cfg.keep_sim_dir:
                shutil.rmtree(sim_dir)

            # Filter non-physical results: delete the whole design folder so the
            # dataset never contains diverged FDTD runs. Reciprocity violations
            # are kept (usually numerical, still roughly usable), but passivity
            # violations (|S|>1) indicate divergence and are always junk.
            if validation is not None and not validation.get("is_passive", False):
                row["status"] = "dropped_not_passive"
                try:
                    shutil.rmtree(workdir)
                except Exception:
                    pass
                return row
            if validation is not None and not validation.get("is_power_passive", False):
                row["status"] = "dropped_not_power_passive"
                try:
                    shutil.rmtree(workdir)
                except Exception:
                    pass
                return row

        # Compute actual fill factor for metadata
        inner = grid[1:-1, 1:-1]
        fill_factor = float(inner.sum()) / inner.size

        meta = {
            "design_id": row["design_id"],
            "seed": seed,
            "inner_size": cfg.inner_size,
            "ports_per_edge": cfg.ports_per_edge,
            "fill_factor": round(fill_factor, 4),
            "em_backend": cfg.em_backend,
            "gds": str(gds_path.relative_to(cfg.output_dir)) if gds_path else "",
            "grid_npy": str(npy_path.relative_to(cfg.output_dir)),
        }
        (workdir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

        row["seconds"] = time.perf_counter() - t0
    except Exception as exc:  # noqa: BLE001
        row["status"] = "error"
        row["error"] = str(exc)
        row["seconds"] = time.perf_counter() - t0

    return row


def run_dataset(cfg: DatasetConfig) -> list[dict]:
    """Generate designs sequentially. For parallel generation, use SLURM array jobs."""
    rows: list[dict] = []
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    for i in range(cfg.num_designs):
        row = generate_one(i, cfg)
        rows.append(row)
        print(
            f"[{i+1}/{cfg.num_designs}] {row['design_id']}: "
            f"{row['status']} ({row['seconds']:.1f}s)",
            file=sys.stderr,
        )
    # Use design_id_offset in manifest filename to avoid collisions in SLURM array jobs
    suffix = f"_{cfg.design_id_offset}" if cfg.design_id_offset > 0 else ""
    manifest = cfg.output_dir / f"manifest{suffix}.csv"
    fieldnames = ["design_id", "seed", "status", "error", "seconds"]
    with manifest.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    # Write validation results separately as JSON for easier parsing
    validations = {
        r["design_id"]: r["validation"]
        for r in rows
        if "validation" in r
    }
    if validations:
        (cfg.output_dir / f"validations{suffix}.json").write_text(
            json.dumps(validations, indent=2), encoding="utf-8",
        )
    summary = cfg.output_dir / f"summary{suffix}.json"
    summary.write_text(
        json.dumps(
            {
                "config": {k: str(v) if isinstance(v, Path) else v for k, v in asdict(cfg).items()},
                "n_total": len(rows),
                "n_ok": sum(1 for r in rows if r.get("status") == "ok"),
                "n_warn": sum(1 for r in rows if r.get("status", "").startswith("warn")),
                "n_err": sum(1 for r in rows if r.get("status") == "error"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return rows
