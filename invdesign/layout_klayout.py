"""Optional KLayout (klayout.db) writer — same geometry as GDSFactory path for cross-check."""

from __future__ import annotations

import json
from pathlib import Path

import klayout.db as pya
import numpy as np

from invdesign.config import LayoutConstants, load_layer_specs

__all__ = ["build_klayout_layout", "write_klayout_gds"]


def build_klayout_layout(
    grid: np.ndarray,
    cell_name: str = "PIXEL_GRID",
    layers: dict[str, tuple[int, int]] | None = None,
    constants: LayoutConstants | None = None,
) -> tuple[pya.Layout, dict[str, str]]:
    layers = layers or load_layer_specs()
    const = constants or LayoutConstants()
    layout = pya.Layout()
    layout.dbu = 0.001  # 1 dbu = 1 nm

    ld_layer = layout.layer(*layers["LD"])
    ld_pins_layer = layout.layer(*layers["LD_pins"])
    m1_layer = layout.layer(*layers["M1_2B"])
    m1_pins_layer = layout.layer(*layers["M1_2B_pins"])
    top = layout.create_cell(cell_name)

    bbox_x0, bbox_y0, _, _ = const.bbox_nm
    origin_x = bbox_x0 + const.margin_nm
    origin_y = bbox_y0 + const.margin_nm
    outer = int(grid.shape[0])
    inner = outer - 2

    port_info: dict[str, list[pya.Box]] = {"top": [], "bottom": [], "left": [], "right": []}

    for i in range(outer):
        for j in range(outer):
            if grid[i, j] != 1:
                continue
            x0 = origin_x + j * const.pixel_step_nm
            y0 = origin_y + (inner + 1 - i) * const.pixel_step_nm
            box = pya.Box(x0, y0, x0 + const.pixel_size_nm, y0 + const.pixel_size_nm)
            top.shapes(ld_layer).insert(box)

            is_top = (i == 0) and (0 < j < outer - 1)
            is_bottom = (i == outer - 1) and (0 < j < outer - 1)
            is_left = (j == 0) and (0 < i < outer - 1)
            is_right = (j == outer - 1) and (0 < i < outer - 1)
            if is_top:
                port_info["top"].append(box)
            elif is_bottom:
                port_info["bottom"].append(box)
            elif is_left:
                port_info["left"].append(box)
            elif is_right:
                port_info["right"].append(box)

    pin_map: dict[str, str] = {}
    port_idx = 0
    for edge, boxes in port_info.items():
        for idx, box in enumerate(sorted(boxes, key=lambda b: (b.p1.x, b.p1.y))):
            port_name = f"{edge.upper()}{idx + 1}"
            pin_id = f"P{port_idx:03d}"
            pin_map[pin_id] = port_name
            top.shapes(ld_pins_layer).insert(box)
            bridge = pya.Box(box.p1.x - 5000, box.p1.y - 5000, box.p2.x + 5000, box.p2.y + 5000)
            top.shapes(m1_layer).insert(bridge)
            top.shapes(m1_pins_layer).insert(bridge)
            port_idx += 1

    return layout, pin_map


def write_klayout_gds(
    grid: np.ndarray,
    out_dir: Path,
    stem: str = "pixel_grid_klayout",
    cell_name: str = "PIXEL_GRID",
) -> tuple[Path, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    layout, pin_map = build_klayout_layout(grid, cell_name=cell_name)
    gds_path = out_dir / f"{stem}.gds"
    layout.write(str(gds_path))
    json_path = out_dir / f"{stem}_pin_map.json"
    json_path.write_text(json.dumps(pin_map, indent=2), encoding="utf-8")
    return gds_path, json_path
