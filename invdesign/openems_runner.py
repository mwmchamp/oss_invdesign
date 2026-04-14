"""OpenEMS helpers: env paths and reference waveguide S-parameter demo."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

from invdesign.touchstone import write_touchstone_nport


def add_paths(install_prefix: Path) -> None:
    """Point OPENEMS/CSXCAD at install prefix and extend LD_LIBRARY_PATH / sys.path."""
    pyver = f"python{sys.version_info.major}.{sys.version_info.minor}"
    os.environ.setdefault("OPENEMS_INSTALL_PATH", str(install_prefix))
    os.environ.setdefault("CSXCAD_INSTALL_PATH", str(install_prefix))
    lib_paths = [install_prefix / "lib", install_prefix / "lib64"]
    ld = ":".join(str(p) for p in lib_paths if p.exists())
    if ld:
        os.environ["LD_LIBRARY_PATH"] = ld + ":" + os.environ.get("LD_LIBRARY_PATH", "")
    site_paths = [
        install_prefix / "lib" / pyver / "site-packages",
        install_prefix / "lib64" / pyver / "site-packages",
    ]
    for p in site_paths:
        if p.exists() and str(p) not in sys.path:
            sys.path.insert(0, str(p))


def run_openems_waveguide(
    out_prefix: Path,
    nr_ts: int = 10_000,
    mesh_div: float = 30.0,
) -> None:
    """
    Reference 2-port rectangular waveguide TE10 sweep (20–26 GHz).

    .. note::
        This does **not** load the pixel GDS; it validates the OpenEMS install and
        produces a plausible .s2p for pipeline tests. Replace with geometry-coupled
        FDTD when your metal stack + ports are modeled in CSXCAD.
    """
    import CSXCAD  # noqa: WPS433
    import openEMS  # noqa: WPS433
    from openEMS.physical_constants import C0  # noqa: WPS433

    unit = 1e-6
    wg_a = 10_700.0
    wg_b = 4_300.0
    wg_len = 50_000.0
    f_start, f_stop = 20e9, 26e9
    f_0 = 24e9
    mesh_res = (C0 / f_0 / unit) / mesh_div

    sim_root = (out_prefix.parent / (out_prefix.name + "_sim")).resolve()
    sim_root.mkdir(parents=True, exist_ok=True)
    run_dir = sim_root / "sim"
    run_dir.mkdir(parents=True, exist_ok=True)

    FDTD = openEMS.openEMS(NrTS=nr_ts, EndCriteria=1e-5)
    FDTD.SetGaussExcite(0.5 * (f_start + f_stop), 0.5 * (f_stop - f_start))
    FDTD.SetBoundaryCond([0, 0, 0, 0, 3, 3])

    CSX = CSXCAD.ContinuousStructure()
    FDTD.SetCSX(CSX)
    mesh = CSX.GetGrid()
    mesh.SetDeltaUnit(unit)
    mesh.AddLine("x", [0, wg_a])
    mesh.AddLine("y", [0, wg_b])
    mesh.AddLine("z", [0, wg_len])

    ports = []
    start = [0, 0, 10 * mesh_res]
    stop = [wg_a, wg_b, 15 * mesh_res]
    mesh.AddLine("z", [start[2], stop[2]])
    ports.append(FDTD.AddRectWaveGuidePort(0, start, stop, "z", wg_a * unit, wg_b * unit, "TE10", 1))

    start = [0, 0, wg_len - 15 * mesh_res]
    stop = [wg_a, wg_b, wg_len - 10 * mesh_res]
    mesh.AddLine("z", [start[2], stop[2]])
    ports.append(FDTD.AddRectWaveGuidePort(1, start, stop, "z", wg_a * unit, wg_b * unit, "TE10"))

    mesh.SmoothMeshLines("all", mesh_res, ratio=1.4)
    FDTD.Run(str(run_dir), cleanup=True)

    freqs = np.linspace(f_start, f_stop, 201)
    for port in ports:
        port.CalcPort(str(run_dir), freqs)

    s11 = ports[0].uf_ref / ports[0].uf_inc
    s21 = ports[1].uf_inc / ports[0].uf_inc

    sparams = np.zeros((len(freqs), 2, 2), dtype=np.complex128)
    sparams[:, 0, 0] = s11
    sparams[:, 1, 0] = s21
    sparams[:, 0, 1] = s21
    sparams[:, 1, 1] = s11

    write_touchstone_nport(out_prefix.with_suffix(".s2p"), 2, freqs, sparams)
    np.save(out_prefix.with_suffix(".freqs.npy"), freqs)
