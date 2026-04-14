"""PNG preview of GDS using gdsfactory (matches notebook visualization style)."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import gdsfactory as gf


def render_gds_png(gds_path: Path, png_path: Path, dpi: int = 200) -> None:
    """Render a GDS file to PNG using gdsfactory's matplotlib backend.

    Same import approach as the EMX notebook:
        gf.clear_cache()
        component = gf.import_gds(gds_path)
        component.plot()
    """
    gds_path = Path(gds_path)
    png_path = Path(png_path)
    png_path.parent.mkdir(parents=True, exist_ok=True)

    gf.clear_cache()
    component = gf.import_gds(str(gds_path))
    component.plot_matplotlib()
    fig = plt.gcf()
    fig.set_size_inches(8, 8)
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
