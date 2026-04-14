"""Command-line interface for invdesign."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from invdesign.dataset import DatasetConfig, run_dataset


def _default_install_prefix() -> Path:
    return Path(os.environ.get("OPENEMS_INSTALL_PATH", "/usr/local"))


def cmd_dataset(args: argparse.Namespace) -> int:
    cfg = DatasetConfig(
        output_dir=Path(args.output).resolve(),
        num_designs=args.num_designs,
        seed_start=args.seed,
        inner_size=args.inner_size,
        ports_per_edge=args.ports_per_edge,
        em_backend=args.em_backend,
        install_prefix=Path(args.install_prefix).resolve(),
        nr_ts=args.nr_ts,
        mesh_div=args.mesh_div,
        write_klayout_copy=args.klayout_copy,
        skip_png=args.skip_png,
        num_ports=args.num_ports,
        f_start=args.f_start,
        f_stop=args.f_stop,
        n_freq=args.n_freq,
        use_pml=not args.no_pml,
        end_criteria_db=args.end_criteria_db,
        xy_pad_um=args.xy_pad_um,
        keep_sim_dir=args.keep_sim_dir,
        design_id_offset=args.design_id_offset,
        connected_grids=getattr(args, "connected_grids", False),
    )
    rows = run_dataset(cfg)
    errors = [r for r in rows if r.get("status") == "error"]
    warns = [r for r in rows if r.get("status", "").startswith("warn")]
    if errors:
        print(f"Completed with {len(errors)} errors, {len(warns)} warnings; see manifest.", file=sys.stderr)
        return 1
    if warns:
        print(f"Wrote {len(rows)} designs ({len(warns)} warnings) under {cfg.output_dir}")
    else:
        print(f"Wrote {len(rows)} designs under {cfg.output_dir}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Pixel-grid RF inverse-design tools (GDSFactory, KLayout, OpenEMS).",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("dataset", help="Generate many design folders (GDS + optional EM).")
    d.add_argument("--output", type=Path, required=True, help="Root output directory (e.g. work_dir/pixelgrid_v1).")
    d.add_argument("--num-designs", type=int, default=10, help="Number of random layouts.")
    d.add_argument("--seed", type=int, default=0, help="Base RNG seed (design i uses seed+i).")
    d.add_argument("--inner-size", type=int, default=25, help="Inner pixel array size (border adds 2).")
    d.add_argument(
        "--ports-per-edge",
        type=int,
        default=1,
        help="Number of port pixels per edge (random placement).",
    )
    d.add_argument("--num-ports", type=int, default=4, help="Number of simulation ports (4 = one per edge).")
    d.add_argument(
        "--em-backend",
        choices=["none", "waveguide", "pixel"],
        default="waveguide",
        help="none=GDS only; waveguide=OpenEMS TE10 wg (not layout); "
        "pixel=CSXCAD model from the same grid as the GDS (slow).",
    )
    d.add_argument("--install-prefix", type=Path, default=_default_install_prefix(), help="openEMS install root.")
    d.add_argument("--nr-ts", type=int, default=1_000_000, help="OpenEMS max timesteps (simulation stops early via EndCriteria).")
    d.add_argument("--mesh-div", type=float, default=20.0, help="Mesh divisor: max cell = lambda_eff/mesh_div (IHP standard = 20).")
    d.add_argument("--f-start", type=float, default=1e9, help="Simulation start frequency in Hz.")
    d.add_argument("--f-stop", type=float, default=30e9, help="Simulation stop frequency in Hz.")
    d.add_argument("--n-freq", type=int, default=30, help="Number of frequency points (1 GHz spacing at 1–30 GHz = 30).")
    d.add_argument("--end-criteria-db", type=float, default=-40.0,
                   help="FDTD convergence criteria in dB (e.g. -35 for PML, -40 for PEC).")
    d.add_argument("--xy-pad-um", type=float, default=150.0, help="XY padding around pixel grid in µm.")
    d.add_argument(
        "--use-pml",
        action="store_true",
        default=True,
        help="Use PML absorbing boundaries (default, more accurate).",
    )
    d.add_argument(
        "--no-pml",
        action="store_true",
        help="Use PEC boundaries instead of PML (faster, less accurate).",
    )
    d.add_argument("--design-id-offset", type=int, default=0,
                   help="Offset for design_XXXXX folder naming (for SLURM array jobs).")
    d.add_argument("--klayout-copy", action="store_true", help="Also write pixel_grid_klayout.gds for diff vs GF.")
    d.add_argument("--skip-png", action="store_true", help="Skip pixel_grid.png rendering.")
    d.add_argument("--keep-sim-dir", action="store_true", help="Keep OpenEMS sim directories (large, ~50 MB each).")
    d.add_argument("--connected-grids", action="store_true",
                   help="Generate grids with random-walk metal paths between ports "
                   "(enriches coupling in training data).")
    d.set_defaults(func=cmd_dataset)

    return p


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
