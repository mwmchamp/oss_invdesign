"""OpenEMS + CSXCAD: 3D voxel model from binary pixel grid using IHP SG13G2 process stack.

Full SG13G2 130 nm SiGe BiCMOS backend with substrate, epi, SiO2, Metal1 ground,
and TopMetal2 pixel grid layer. All physical layers are modeled for fidelity.

**Design decisions for dataset generation (documented):**

1. **Full stack retained**: Substrate (ε=11.9, σ=2 S/m), epi (ε=11.9, σ=5 S/m),
   SiO2 (ε=4.1) are all modeled because:
   - Substrate losses affect Q-factor and S-parameter magnitudes
   - Epi layer conductivity provides realistic RF ground return path
   - Removing these layers produces non-passive S-parameters (|S| > 1 artifacts)

2. **Domain sizing**: XY padding reduced from 1× geometry to 100 µm.
   With PEC boundaries, large padding just creates a larger cavity with more
   resonances. 100 µm provides enough margin for fringe fields without
   excessive computation. Air above reduced from 300 µm to 100 µm.

3. **Substrate mesh**: Coarse exponential meshing in substrate (10 cells)
   rather than fine uniform mesh. The substrate is below the PEC ground plane
   and acts primarily as a lossy medium — fine spatial resolution isn't needed.

4. **Boundaries**: PEC on all faces (default). This makes the structure a
   shielded cavity, consistent across all designs. For CNN training, consistency
   matters more than absolute accuracy vs measurement.

Stack parameters imported from IHP-Open-PDK:
  IHP-Open-PDK/ihp-sg13g2/libs.tech/openems/openems_ihp_sg13g2/
"""

from __future__ import annotations

import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from invdesign.config import LayoutConstants
from invdesign.layout_geom import cell_rect_um, four_port_rects_um, layout_xy_extents_um
from invdesign.touchstone import write_touchstone_nport

# Import process stack from IHP-Open-PDK
_PDK_OPENEMS = Path(__file__).resolve().parent.parent.parent / "IHP-Open-PDK" / "ihp-sg13g2" / "libs.tech" / "openems"
if str(_PDK_OPENEMS) not in sys.path:
    sys.path.insert(0, str(_PDK_OPENEMS))

from openems_ihp_sg13g2 import SG13G2Stack as _PDKStack


# ---------------------------------------------------------------------------
# SG13G2 process stack — thin adapter over PDK-sourced data
# ---------------------------------------------------------------------------

class SG13G2Stack:
    """IHP SG13G2 stack adapter for pixel grid simulation.

    Wraps the canonical stack definition from IHP-Open-PDK, exposing only the
    layers used in pixel grid simulation (substrate, epi, SiO2, Metal1, TopMetal2).
    Intermediate metals (Metal2-5, TopMetal1) and vias are available via self.pdk.
    """

    def __init__(self, pdk: _PDKStack | None = None):
        self.pdk = pdk or _PDKStack()

    # --- Substrate ---
    @property
    def sub_thick_um(self) -> float:
        return self.pdk.substrate.thickness_um

    @property
    def sub_eps(self) -> float:
        return self.pdk.substrate.epsilon

    @property
    def sub_kappa(self) -> float:
        return self.pdk.substrate.kappa

    # --- Epitaxial ---
    @property
    def epi_thick_um(self) -> float:
        return self.pdk.epitaxial.thickness_um

    @property
    def epi_eps(self) -> float:
        return self.pdk.epitaxial.epsilon

    @property
    def epi_kappa(self) -> float:
        return self.pdk.epitaxial.kappa

    # --- SiO2 ---
    @property
    def sio2_thick_um(self) -> float:
        return self.pdk.sio2.thickness_um

    @property
    def sio2_eps(self) -> float:
        return self.pdk.sio2.epsilon

    # --- Metal1 (ground plane) ---
    @property
    def m1_offset_um(self) -> float:
        return self.pdk.metal1.offset_um

    @property
    def m1_thick_um(self) -> float:
        return self.pdk.metal1.thickness_um

    # --- TopMetal2 (pixel grid layer) ---
    @property
    def tm2_offset_um(self) -> float:
        return self.pdk.topmetal2.offset_um

    @property
    def tm2_thick_um(self) -> float:
        return self.pdk.topmetal2.thickness_um

    @property
    def tm2_sigma(self) -> float:
        return self.pdk.topmetal2.sigma

    # --- Air ---
    @property
    def air_thick_um(self) -> float:
        return self.pdk.air_thickness_um

    # --- Absolute z positions ---

    @property
    def sub_zmin(self) -> float:
        return self.pdk.sub_zmin

    @property
    def sub_zmax(self) -> float:
        return self.pdk.sub_zmax

    @property
    def epi_zmin(self) -> float:
        return self.pdk.epi_zmin

    @property
    def epi_zmax(self) -> float:
        return self.pdk.epi_zmax

    @property
    def sio2_zmin(self) -> float:
        return self.pdk.sio2_zmin

    @property
    def sio2_zmax(self) -> float:
        return self.pdk.sio2_zmax

    @property
    def m1_zmin(self) -> float:
        return self.pdk.metal_z("Metal1")[0]

    @property
    def m1_zmax(self) -> float:
        return self.pdk.metal_z("Metal1")[1]

    @property
    def tm2_zmin(self) -> float:
        return self.pdk.metal_z("TopMetal2")[0]

    @property
    def tm2_zmax(self) -> float:
        return self.pdk.metal_z("TopMetal2")[1]

    @property
    def air_zmax(self) -> float:
        return self.pdk.air_zmax


def validate_sparams(
    sparams: np.ndarray,
    *,
    passivity_tol: float = 0.01,
    reciprocity_tol: float = 0.05,
) -> dict:
    """Check passivity and reciprocity for an N-port S-parameter matrix.

    Parameters
    ----------
    sparams : ndarray, shape (F, N, N)

    Returns a dict with boolean flags and numeric diagnostics.
    """
    n_ports = sparams.shape[1]
    mags = np.abs(sparams)

    # Element-wise passivity: |Sij| ≤ 1
    max_mag = float(mags.max())
    all_passive = max_mag <= 1.0 + passivity_tol

    # Power conservation: sum_i |S_ij|^2 ≤ 1 for each column j at each freq
    power_per_col = (mags ** 2).sum(axis=1)  # (F, N)
    max_power = float(power_per_col.max())
    power_passive = max_power <= 1.0 + passivity_tol

    # Reciprocity: Sij ≈ Sji for all i≠j pairs
    max_recip = 0.0
    recip_errors = []
    for i in range(n_ports):
        for j in range(i + 1, n_ports):
            err = np.abs(sparams[:, i, j] - sparams[:, j, i])
            norm = np.maximum(np.abs(sparams[:, i, j]), np.abs(sparams[:, j, i])) + 1e-10
            rel_err = err / norm
            max_recip = max(max_recip, float(err.max()))
            recip_errors.append(float(rel_err.mean()))
    mean_recip_rel = float(np.mean(recip_errors)) if recip_errors else 0.0
    is_reciprocal = mean_recip_rel <= reciprocity_tol

    return {
        "n_ports": n_ports,
        "is_passive": all_passive,
        "is_power_passive": power_passive,
        "is_reciprocal": is_reciprocal,
        "max_mag": max_mag,
        "max_power_per_col": max_power,
        "max_reciprocity_error": max_recip,
        "mean_reciprocity_rel_error": mean_recip_rel,
    }


def run_openems_pixel_grid(
    out_prefix: Path,
    grid: np.ndarray,
    *,
    const: LayoutConstants | None = None,
    stack: SG13G2Stack | None = None,
    nr_ts: int = 1_000_000,
    mesh_div: float = 20.0,
    num_ports: int = 4,
    lumped_r: float = 50.0,
    f_start: float = 1e9,
    f_stop: float = 30e9,
    n_freq: int = 30,
    end_criteria_db: float = -40.0,
    use_pml: bool = True,
    xy_pad_um: float = 200.0,
    antenna_mode: bool = False,
    nf2ff: bool = False,
) -> dict | None:
    """Build SG13G2 stack with pixel grid on TopMetal2.

    When antenna_mode=False (default): Metal1 PEC ground plane present.
    4 lumped ports (one per edge: N=0, S=1, E=2, W=3) connect Metal1 ground
    to TopMetal2 signal (following IHP reference methodology).

    Domain sizing decisions (for speed without sacrificing fidelity):
      - XY: pixel grid extent + xy_pad_um (100 µm default)
      - Z: full substrate + epi + SiO2 + air, but substrate uses coarse mesh
      - Substrate mesh: ~10 exponentially-spaced cells (vs ~200 uniform)
      - Air above: 100 µm (vs 300 µm in IHP reference — acceptable with PEC cap)

    Returns validation dict if simulation succeeds, None on error.
    """
    import CSXCAD
    import openEMS
    from openEMS.physical_constants import C0

    out_prefix = Path(out_prefix).resolve()
    const = const or LayoutConstants()
    stk = stack or SG13G2Stack()
    grid = np.asarray(grid, dtype=np.int8)
    unit = 1e-6
    f_c = 0.5 * (f_start + f_stop)

    pixel_um = const.pixel_size_nm * 1e-3
    step_um = const.pixel_step_nm * 1e-3

    # Mesh resolution: IHP reference methodology
    lam_eff_um = (C0 / f_c / unit) / np.sqrt(stk.sio2_eps)
    mesh_res = min(lam_eff_um / mesh_div, pixel_um / 4.0, step_um / 4.0)
    mesh_res = max(mesh_res, 1.0)

    wavelength_air_um = (C0 / f_stop / unit)
    max_cellsize = wavelength_air_um / (np.sqrt(stk.sub_eps) * 20)

    # XY domain: pixel grid extent + moderate padding
    xmin, ymin, xmax, ymax = layout_xy_extents_um(grid, const)
    if use_pml:
        pml_depth = 8 * max_cellsize * 1.5  # generous margin for PML region
        pad = max(pml_depth + 150.0, xy_pad_um)
    else:
        pad = xy_pad_um
    xmin -= pad
    xmax += pad
    ymin -= pad
    ymax += pad

    pml_gap = 50.0 if use_pml else 0.0

    # Port geometry — one port per edge
    port_rects = four_port_rects_um(grid, const)
    port_order = ["north", "south", "east", "west"]

    sim_root = (out_prefix.parent / (out_prefix.name + "_pixel_sim")).resolve()
    sim_root.mkdir(parents=True, exist_ok=True)
    freqs = np.linspace(f_start, f_stop, n_freq)

    def _port_start_stop(edge: str, rect: tuple[float, float, float, float]):
        """Return (start, stop, direction) for a vertical lumped port."""
        x0, y0, x1, y1 = rect
        z_lo = stk.m1_zmax
        z_hi = stk.tm2_zmin
        if edge == "north":
            return [x0, y1, z_lo], [x1, y1, z_hi], "z"
        elif edge == "south":
            return [x0, y0, z_lo], [x1, y0, z_hi], "z"
        elif edge == "east":
            return [x1, y0, z_lo], [x1, y1, z_hi], "z"
        elif edge == "west":
            return [x0, y0, z_lo], [x0, y1, z_hi], "z"
        raise ValueError(f"Unknown edge: {edge}")

    def build_and_run(excite_idx: int) -> list:
        """Build geometry and run FDTD with one port excited."""
        run_dir = sim_root / f"excite_{excite_idx}"
        if run_dir.exists():
            shutil.rmtree(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)

        lim = 10.0 ** (end_criteria_db / 10.0)
        FDTD = openEMS.openEMS(NrTS=nr_ts, EndCriteria=lim)
        FDTD.SetGaussExcite(0.5 * (f_start + f_stop), 0.5 * (f_stop - f_start))
        if antenna_mode:
            # Antenna: PML on all 6 faces for radiation
            FDTD.SetBoundaryCond(["PML_8"] * 6)
        elif use_pml:
            FDTD.SetBoundaryCond(["PML_8", "PML_8", "PML_8", "PML_8", "PEC", "PML_8"])
        else:
            FDTD.SetBoundaryCond(["PEC", "PEC", "PEC", "PEC", "PEC", "PEC"])

        CSX = CSXCAD.ContinuousStructure()
        FDTD.SetCSX(CSX)
        mesh = CSX.GetGrid()
        mesh.SetDeltaUnit(unit)

        # --- SG13G2 material definitions ---
        sub_mat = CSX.AddMaterial("substrate", epsilon=stk.sub_eps, kappa=stk.sub_kappa)
        epi_mat = CSX.AddMaterial("epi", epsilon=stk.epi_eps, kappa=stk.epi_kappa)
        sio2_mat = CSX.AddMaterial("SiO2", epsilon=stk.sio2_eps)
        tm2_mat = CSX.AddMaterial("TopMetal2", kappa=stk.tm2_sigma)
        m1_pec = CSX.AddMetal("Metal1_GND")

        # --- Bulk layers ---
        bx0 = xmin + pml_gap
        bx1 = xmax - pml_gap
        by0 = ymin + pml_gap
        by1 = ymax - pml_gap
        sub_mat.AddBox([bx0, by0, stk.sub_zmin], [bx1, by1, stk.sub_zmax], priority=10)
        epi_mat.AddBox([bx0, by0, stk.epi_zmin], [bx1, by1, stk.epi_zmax], priority=10)
        sio2_mat.AddBox([bx0, by0, stk.sio2_zmin], [bx1, by1, stk.sio2_zmax], priority=10)

        # PEC ground at Metal1 top surface (omitted in antenna mode)
        m1_z = stk.m1_zmax
        if not antenna_mode:
            m1_pec.AddBox([bx0, by0, m1_z], [bx1, by1, m1_z], priority=100)

        # TopMetal2 pixel grid
        outer = int(grid.shape[0])
        for i in range(outer):
            for j in range(outer):
                if grid[i, j] != 1:
                    continue
                x0, y0, x1, y1 = cell_rect_um(i, j, outer, const)
                tm2_mat.AddBox(
                    [x0, y0, stk.tm2_zmin],
                    [x1, y1, stk.tm2_zmax],
                    priority=200,
                )

        # --- Mesh: XY ---
        # Boundary lines
        mesh.AddLine("x", xmin)
        mesh.AddLine("x", xmax)
        mesh.AddLine("y", ymin)
        mesh.AddLine("y", ymax)

        # Pixel edge mesh lines (conductor pixels only)
        for i in range(outer):
            for j in range(outer):
                if grid[i, j] != 1:
                    continue
                x0, y0, x1, y1 = cell_rect_um(i, j, outer, const)
                mesh.AddLine("x", x0)
                mesh.AddLine("x", x1)
                mesh.AddLine("y", y0)
                mesh.AddLine("y", y1)

        # Port mesh lines and port definitions
        ports = []
        for pidx, edge in enumerate(port_order):
            rect = port_rects[edge]
            px0, py0, px1, py1 = rect
            mesh.AddLine("x", px0)
            mesh.AddLine("x", px1)
            mesh.AddLine("y", py0)
            mesh.AddLine("y", py1)

            start, stop, direction = _port_start_stop(edge, rect)
            port = FDTD.AddLumpedPort(
                pidx, lumped_r,
                start, stop, direction,
                excite=1.0 if pidx == excite_idx else 0.0,
                priority=250,
            )
            ports.append(port)

        # --- Mesh: Z ---
        # Substrate: coarse exponential mesh (10 cells)
        # The substrate is lossy and below the ground plane — fine resolution
        # isn't needed, but it must be present for correct capacitive coupling.
        n_sub = 10
        # Exponential spacing: finer near top (epi interface), coarser at bottom
        ratios = np.geomspace(1.0, 8.0, n_sub)
        ratios = ratios / ratios.sum() * stk.sub_thick_um
        z = stk.sub_zmax
        for dz in ratios:
            mesh.AddLine("z", z)
            z -= dz
        mesh.AddLine("z", stk.sub_zmin)

        # Epi layer: 2-3 cells
        n_epi = max(2, int(np.ceil(stk.epi_thick_um / mesh_res)))
        for z in np.linspace(stk.epi_zmin, stk.epi_zmax, n_epi + 1):
            mesh.AddLine("z", float(z))

        # SiO2: key region — Metal1 ground to TopMetal2, needs adequate resolution
        mesh.AddLine("z", stk.sio2_zmin)
        mesh.AddLine("z", m1_z)  # Metal1 top surface
        n_oxide = max(4, int(np.ceil((stk.tm2_zmin - m1_z) / mesh_res)))
        for z in np.linspace(m1_z, stk.tm2_zmin, n_oxide + 1):
            mesh.AddLine("z", float(z))

        # TopMetal2: 2-3 cells for the 3 µm thick layer
        n_tm2 = max(2, int(np.ceil(stk.tm2_thick_um / mesh_res)))
        for z in np.linspace(stk.tm2_zmin, stk.tm2_zmax, n_tm2 + 1):
            mesh.AddLine("z", float(z))

        mesh.AddLine("z", stk.sio2_zmax)

        # Air above: expanding mesh
        # PML needs ≥8 buffer cells between SiO2 top and PML region.
        # Start finer (mesh_res) and grow gently so PML_8 doesn't start
        # right at the material interface (causes divergence).
        z = stk.sio2_zmax
        step = mesh_res if use_pml else 2.0 * mesh_res
        growth = 1.2 if use_pml else 1.3
        while z < stk.air_zmax:
            mesh.AddLine("z", z)
            z += step
            step = min(step * growth, max_cellsize)
        mesh.AddLine("z", stk.air_zmax)

        # Smooth mesh — limit growth ratio to 1.2 for PML stability
        smooth_ratio = 1.2 if use_pml else 1.3
        mesh.SmoothMeshLines("x", max_cellsize, smooth_ratio)
        mesh.SmoothMeshLines("y", max_cellsize, smooth_ratio)

        # NF2FF recording box for antenna far-field calculation
        nf2ff_box = None
        if nf2ff and antenna_mode:
            nf2ff_box = FDTD.CreateNF2FFBox()

        FDTD.Run(str(run_dir), cleanup=True)
        for p in ports:
            p.CalcPort(str(run_dir), freqs, ref_impedance=lumped_r)
        return ports, nf2ff_box

    # Run N excitations (one per port) to build full S-matrix
    n_ports = num_ports
    sparams = np.zeros((len(freqs), n_ports, n_ports), dtype=np.complex128)

    last_nf2ff = None
    for excite_idx in range(n_ports):
        result = build_and_run(excite_idx)
        ports, nf2ff_box = result
        if nf2ff_box is not None:
            last_nf2ff = nf2ff_box
        excited_port = ports[excite_idx]
        for j in range(n_ports):
            sparams[:, j, excite_idx] = ports[j].uf_ref / excited_port.uf_inc

    ext = f".s{n_ports}p"
    write_touchstone_nport(out_prefix.with_suffix(ext), n_ports, freqs, sparams)
    np.save(out_prefix.with_suffix(".freqs.npy"), freqs)

    return validate_sparams(sparams)
