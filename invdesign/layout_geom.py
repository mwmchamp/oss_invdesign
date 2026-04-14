"""Shared pixel-grid geometry in micrometers (matches ``layout_gf`` conventions)."""

from __future__ import annotations

import numpy as np

from invdesign.config import LayoutConstants

NM_TO_UM = 1e-3


def layout_origin_nm(const: LayoutConstants) -> tuple[float, float]:
    bbox_x0, bbox_y0, _, _ = const.bbox_nm
    return float(bbox_x0 + const.margin_nm), float(bbox_y0 + const.margin_nm)


def cell_rect_um(i: int, j: int, outer: int, const: LayoutConstants) -> tuple[float, float, float, float]:
    """Return (x0, y0, x1, y1) in µm for cell ``(i, j)`` (row, col), same as GDSFactory path."""
    inner = outer - 2
    ox, oy = layout_origin_nm(const)
    x0_nm = ox + j * const.pixel_step_nm
    y0_nm = oy + (inner + 1 - i) * const.pixel_step_nm
    x1_nm = x0_nm + const.pixel_size_nm
    y1_nm = y0_nm + const.pixel_size_nm
    return (
        x0_nm * NM_TO_UM,
        y0_nm * NM_TO_UM,
        x1_nm * NM_TO_UM,
        y1_nm * NM_TO_UM,
    )


def sorted_port_rects_um(
    grid: np.ndarray,
    const: LayoutConstants | None = None,
) -> dict[str, list[tuple[float, float, float, float]]]:
    """Edge-keyed port pixel rectangles (µm), sorted like ``build_pixel_component``."""
    const = const or LayoutConstants()
    outer = int(grid.shape[0])
    port_boxes: dict[str, list[tuple[float, float, float, float]]] = {
        "top": [],
        "bottom": [],
        "left": [],
        "right": [],
    }
    for i in range(outer):
        for j in range(outer):
            if grid[i, j] != 1:
                continue
            x0, y0, x1, y1 = cell_rect_um(i, j, outer, const)
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
    return {edge: sorted(boxes, key=lambda t: (t[0], t[1])) for edge, boxes in port_boxes.items()}


def default_top_bottom_port_rects_um(
    grid: np.ndarray,
    const: LayoutConstants | None = None,
) -> tuple[tuple[float, float, float, float], tuple[float, float, float, float]]:
    """First top-edge and first bottom-edge port rectangles (for a 2-port CSXCAD model)."""
    const = const or LayoutConstants()
    pb = sorted_port_rects_um(grid, const)
    top = pb.get("top") or []
    bottom = pb.get("bottom") or []
    if not top:
        raise ValueError("Pixel grid has no top-edge port cells (row 0, inner columns).")
    if not bottom:
        raise ValueError("Pixel grid has no bottom-edge port cells (last row, inner columns).")
    return top[0], bottom[0]


def four_port_rects_um(
    grid: np.ndarray,
    const: LayoutConstants | None = None,
) -> dict[str, tuple[float, float, float, float]]:
    """Return exactly one port rect per edge for 4-port simulation (N, S, E, W).

    Expects one border port pixel per edge (centered or random). If multiple port
    pixels exist on an edge, uses the middle entry after sorting.
    Returns dict with keys 'north', 'south', 'east', 'west'.
    """
    const = const or LayoutConstants()
    pb = sorted_port_rects_um(grid, const)
    result: dict[str, tuple[float, float, float, float]] = {}
    for edge, key in [("top", "north"), ("bottom", "south"), ("left", "west"), ("right", "east")]:
        boxes = pb.get(edge) or []
        if not boxes:
            raise ValueError(f"Pixel grid has no {edge}-edge port cells.")
        # Take the center of the sorted list (only entry when one port per edge)
        result[key] = boxes[len(boxes) // 2]
    return result


def layout_xy_extents_um(grid: np.ndarray, const: LayoutConstants | None = None) -> tuple[float, float, float, float]:
    """Bounding box (xmin, ymin, xmax, ymax) in µm over all conductor pixels."""
    const = const or LayoutConstants()
    outer = int(grid.shape[0])
    xs: list[float] = []
    ys: list[float] = []
    for i in range(outer):
        for j in range(outer):
            if grid[i, j] != 1:
                continue
            x0, y0, x1, y1 = cell_rect_um(i, j, outer, const)
            xs.extend((x0, x1))
            ys.extend((y0, y1))
    if not xs:
        raise ValueError("Grid has no conductor pixels.")
    return min(xs), min(ys), max(xs), max(ys)
