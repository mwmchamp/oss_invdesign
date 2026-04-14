"""Geometry and layer defaults (IHP SG13G2–style GDS layers)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

_DEFAULT_TECH = Path(__file__).resolve().parent.parent / "openems_flow" / "tech" / "ihp_example_layers.yaml"


@dataclass(frozen=True)
class LayoutConstants:
    """Physical layout in GDS database units: nanometers (1 dbu = 1 nm when scale is set accordingly)."""

    bbox_nm: tuple[int, int, int, int] = (-100_000, -100_000, 1_075_000, 1_075_000)
    pixel_size_nm: int = 47_500
    pixel_step_nm: int = 38_000
    margin_nm: int = 95_250


def load_layer_specs(yaml_path: Path | None = None) -> dict[str, tuple[int, int]]:
    """Load layer (layer, datatype) tuples from tech YAML."""
    path = yaml_path or _DEFAULT_TECH
    with path.open() as f:
        data: dict[str, Any] = yaml.safe_load(f)
    out: dict[str, tuple[int, int]] = {}
    for name, spec in data.get("layers", {}).items():
        out[name] = (int(spec["layer"]), int(spec["datatype"]))
    return out
