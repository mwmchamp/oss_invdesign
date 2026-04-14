"""Build pixel layouts with GDSFactory (primary GDS path)."""

from __future__ import annotations

import json
from pathlib import Path

import gdsfactory as gf
import numpy as np

from invdesign.config import LayoutConstants, load_layer_specs

NM_TO_UM = 1e-3


def build_pixel_component(
    grid: np.ndarray,
    cell_name: str = "PIXEL_GRID",
    layers: dict[str, tuple[int, int]] | None = None,
    constants: LayoutConstants | None = None,
) -> tuple[gf.Component, dict[str, str]]:
    """
    Return a gdsfactory Component and P000-style pin map (matches legacy KLayout flow).
    Coordinates follow the original script (nm → µm for GDSFactory).
    """
    layers = layers or load_layer_specs()
    const = constants or LayoutConstants()
    ld = layers["LD"]
    ld_pins = layers["LD_pins"]
    m1 = layers["M1_2B"]
    m1_pins = layers["M1_2B_pins"]

    bbox_x0, bbox_y0, _, _ = const.bbox_nm
    origin_x = bbox_x0 + const.margin_nm
    origin_y = bbox_y0 + const.margin_nm

    c = gf.Component(name=cell_name)
    outer = int(grid.shape[0])
    inner = outer - 2

    port_boxes: dict[str, list[tuple[float, float, float, float]]] = {
        "top": [],
        "bottom": [],
        "left": [],
        "right": [],
    }

    def nm_rect_um(i: int, j: int) -> tuple[float, float, float, float]:
        x0_nm = origin_x + j * const.pixel_step_nm
        y0_nm = origin_y + (inner + 1 - i) * const.pixel_step_nm
        x1_nm = x0_nm + const.pixel_size_nm
        y1_nm = y0_nm + const.pixel_size_nm
        return (
            x0_nm * NM_TO_UM,
            y0_nm * NM_TO_UM,
            x1_nm * NM_TO_UM,
            y1_nm * NM_TO_UM,
        )

    for i in range(outer):
        for j in range(outer):
            if grid[i, j] != 1:
                continue
            x0, y0, x1, y1 = nm_rect_um(i, j)
            c.add_polygon([(x0, y0), (x1, y0), (x1, y1), (x0, y1)], layer=ld)

            is_top = (i == 0) and (0 < j < outer - 1)
            is_bottom = (i == outer - 1) and (0 < j < outer - 1)
            is_left = (j == 0) and (0 < i < outer - 1)
            is_right = (j == outer - 1) and (0 < i < outer - 1)
            if is_top:
                port_boxes["top"].append((x0, y0, x1, y1))
            elif is_bottom:
                port_boxes["bottom"].append((x0, y0, x1, y1))
            elif is_left:
                port_boxes["left"].append((x0, y0, x1, y1))
            elif is_right:
                port_boxes["right"].append((x0, y0, x1, y1))

    pin_map: dict[str, str] = {}
    port_idx = 0
    um5 = 5.0  # 5000 nm margin for bridge
    for edge in ("top", "bottom", "left", "right"):
        boxes = sorted(port_boxes[edge], key=lambda t: (t[0], t[1]))
        for idx_box, (x0, y0, x1, y1) in enumerate(boxes):
            port_name = f"{edge.upper()}{idx_box + 1}"
            pin_id = f"P{port_idx:03d}"
            pin_map[pin_id] = port_name
            c.add_polygon([(x0, y0), (x1, y0), (x1, y1), (x0, y1)], layer=ld_pins)
            bx0, by0 = x0 - um5, y0 - um5
            bx1, by1 = x1 + um5, y1 + um5
            c.add_polygon([(bx0, by0), (bx1, by0), (bx1, by1), (bx0, by1)], layer=m1)
            c.add_polygon([(bx0, by0), (bx1, by0), (bx1, by1), (bx0, by1)], layer=m1_pins)
            port_idx += 1

    return c, pin_map


def write_layout_bundle(
    grid: np.ndarray,
    out_dir: Path,
    stem: str = "pixel_grid",
    cell_name: str = "PIXEL_GRID",
) -> tuple[Path, Path, Path]:
    """
    Write ``stem.gds``, ``stem_pin_map.json``, ``stem.npy`` under ``out_dir``.
    Returns paths (gds, json, npy).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    comp, pin_map = build_pixel_component(grid, cell_name=cell_name)
    gds_path = out_dir / f"{stem}.gds"
    comp.write_gds(gds_path)
    json_path = out_dir / f"{stem}_pin_map.json"
    json_path.write_text(json.dumps(pin_map, indent=2), encoding="utf-8")
    npy_path = out_dir / f"{stem}.npy"
    np.save(npy_path, grid.astype(np.int8))
    return gds_path, json_path, npy_path
