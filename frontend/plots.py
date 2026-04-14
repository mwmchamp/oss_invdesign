"""Matplotlib figures for the frontend results panel.

All functions return a matplotlib Figure so Gradio's `gr.Plot` can render it.
"""

from __future__ import annotations

import numpy as np

from optimizer.objectives import ImpedanceMatchObjective
from frontend.goals_io import PORT_LABEL

_BG = "#1a1a2e"
_PANEL = "#16213e"
_EDGE = "#2f2f48"
_TEXT = "#e0e0e8"

_PAIR_COLORS = {
    (0, 0): "#e74c3c", (1, 1): "#c0392b", (2, 2): "#d35400", (3, 3): "#a04000",
    (0, 1): "#2980b9", (1, 0): "#2980b9",
    (0, 2): "#27ae60", (2, 0): "#27ae60",
    (0, 3): "#8e44ad", (3, 0): "#8e44ad",
    (1, 2): "#f39c12", (2, 1): "#f39c12",
    (1, 3): "#1abc9c", (3, 1): "#1abc9c",
    (2, 3): "#e67e22", (3, 2): "#e67e22",
}


def _dark(ax, fig):
    fig.patch.set_facecolor(_BG)
    ax.set_facecolor(_PANEL)
    ax.tick_params(colors="#c0c0d0")
    for sp in ax.spines.values():
        sp.set_color(_EDGE)
    ax.grid(True, alpha=0.2, color="#4a4a68")


def plot_grid(grid: np.ndarray):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    rows, cols = grid.shape
    fill = grid[1:-1, 1:-1].mean()
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    _dark(ax, fig)
    ax.set_facecolor(_PANEL)
    for r in range(rows):
        for c in range(cols):
            if grid[r, c]:
                border = (r in (0, rows - 1)) or (c in (0, cols - 1))
                color = "#e94560" if border else "#f0c040"
                ax.add_patch(Rectangle((c - 0.45, r - 0.45), 0.9, 0.9,
                                        facecolor=color, edgecolor="#0f3460",
                                        linewidth=0.4, alpha=0.95))
    for label, (r, c) in [("N", (-0.9, cols / 2 - 0.5)),
                           ("S", (rows - 0.1, cols / 2 - 0.5)),
                           ("W", (rows / 2 - 0.5, -0.9)),
                           ("E", (rows / 2 - 0.5, cols - 0.1))]:
        ax.text(c, r, label, color="#e94560", fontsize=11,
                fontweight="bold", ha="center", va="center")
    ax.set_xlim(-1.5, cols + 0.5); ax.set_ylim(rows + 0.5, -1.5)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"Best design · {rows}×{cols} · fill {fill:.0%}",
                 color="white", fontsize=10, pad=6)
    import matplotlib.pyplot as _plt
    _plt.tight_layout(pad=0.3)
    return fig


def plot_sparams(sparams: np.ndarray, objective):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    freqs = objective.freq_ghz
    f0, f1 = float(freqs[0]), float(freqs[-1])
    fig, ax = plt.subplots(figsize=(7, 3.8))
    _dark(ax, fig)

    if isinstance(objective, ImpedanceMatchObjective):
        sparam_goals = list(objective.sparam_goals)
        pairs = set()
        for g in objective.goals:
            pairs.add((g.in_port, g.out_port))
            pairs.add((g.in_port, g.in_port))
        for g in sparam_goals:
            pairs.add((g.i, g.j))
    else:
        sparam_goals = list(objective.goals)
        pairs = {(g.i, g.j) for g in sparam_goals}

    styles = ["-", "--", "-.", ":"]
    for idx, pair in enumerate(sorted(pairs)):
        mag_db = 20 * np.log10(np.abs(sparams[:, pair[0], pair[1]]) + 1e-12)
        ax.plot(freqs, mag_db,
                color=_PAIR_COLORS.get(pair, "#95a5a6"),
                linestyle=styles[idx % len(styles)], lw=1.8,
                label=f"|S{PORT_LABEL[pair[0]]}{PORT_LABEL[pair[1]]}|")

    for g in sparam_goals:
        color = {"above": "#27ae60", "below": "#e74c3c",
                 "at": "#f1c40f"}.get(g.mode, "#95a5a6")
        xmin = (g.f_min_ghz - f0) / max(f1 - f0, 1e-9)
        xmax = (g.f_max_ghz - f0) / max(f1 - f0, 1e-9)
        if g.mode == "above":
            ax.axhspan(g.target_db, 5, alpha=0.08, color=color,
                       xmin=xmin, xmax=xmax)
        elif g.mode == "below":
            ax.axhspan(-80, g.target_db, alpha=0.06, color=color,
                       xmin=xmin, xmax=xmax)
        ax.hlines(g.target_db, g.f_min_ghz, g.f_max_ghz,
                  colors=color, linewidth=1.2, linestyle=":")

    ax.set_xlabel("Frequency (GHz)", color=_TEXT, fontsize=10)
    ax.set_ylabel("|S| (dB)", color=_TEXT, fontsize=10)
    ax.set_xlim(f0, f1); ax.set_ylim(-60, 5)
    ax.legend(loc="lower left", ncol=3, fontsize=8,
              facecolor=_BG, edgecolor=_EDGE, labelcolor=_TEXT)
    import matplotlib.pyplot as _plt
    _plt.tight_layout(pad=0.3)
    return fig


def plot_convergence(history_best, history_mean):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 2.5))
    _dark(ax, fig)
    gens = range(len(history_best))
    ax.plot(gens, history_best, color="#27ae60", lw=1.8, label="Best")
    ax.plot(gens, history_mean, color="#f39c12", lw=1.2, alpha=0.8, label="Mean")
    ax.set_xlabel("Generation", color=_TEXT, fontsize=9)
    ax.set_ylabel("Fitness", color=_TEXT, fontsize=9)
    ax.legend(loc="lower right", fontsize=8,
              facecolor=_BG, edgecolor=_EDGE, labelcolor=_TEXT)
    import matplotlib.pyplot as _plt
    _plt.tight_layout(pad=0.3)
    return fig
