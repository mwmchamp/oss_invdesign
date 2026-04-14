"""Slow-path operations invoked by UI click handlers.

Keeps the heavy lifting (GA, active learning, LLM parse, export) out of
`frontend.app` so that UI wiring stays short and readable.

Progress semantics: handlers that take >1 s accept a `progress=gr.Progress()`
callable. Fast handlers (parse, preset, upload) do NOT use gr.Progress — in
Gradio 6 a handler that finishes in <100 ms while firing multiple SSE progress
frames can desync the browser WebSocket and freeze the UI.
"""

from __future__ import annotations

import json
import tempfile
import time
import zipfile
from datetime import datetime
from pathlib import Path

import gradio as gr
import numpy as np

from frontend import state
from frontend.goals_io import (
    GOAL_COLUMNS, PORT_LABEL, build_objective, df_to_impedance_goals,
    empty_goals_df, goals_to_df,
)
from frontend.llm_extract import extract_objective, PromptRejectedError
from frontend.plots import plot_grid, plot_sparams, plot_convergence
from optimizer.objectives import ImpedanceMatchObjective, MatchingObjective


def _fig_to_png(fig, stem: str) -> str:
    """Save a Matplotlib Figure to a PNG tempfile and return its path.

    Returning paths (not Figure objects) lets Gradio render via `gr.Image`,
    which provides the built-in lightbox (click-to-expand) and download
    button the user requested. `gr.Plot` in Gradio 6 supports neither.
    """
    import matplotlib.pyplot as plt
    tmp = Path(tempfile.mkdtemp(prefix="rfplot_")) / f"{stem}.png"
    fig.savefig(tmp, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    return str(tmp)


def _sparam_compass(label: str) -> str:
    """Rewrite 'S21' / '|S11|' → 'S_SN' / '|S_NN|' using compass directions."""
    import re
    def _sub(m):
        i = int(m.group(1)) - 1
        j = int(m.group(2)) - 1
        return f"S_{PORT_LABEL.get(i, i)}{PORT_LABEL.get(j, j)}"
    return re.sub(r"S(\d)(\d)", _sub, label)


def _objective_to_json(obj) -> dict:
    """Serialize a MatchingObjective or ImpedanceMatchObjective to spec JSON."""
    data = {"name": getattr(obj, "name", "design"),
            "description": getattr(obj, "description", ""),
            "sparam_goals": [], "impedance_goals": []}
    sgs = []
    if isinstance(obj, ImpedanceMatchObjective):
        sgs = list(obj.sparam_goals)
        for g in obj.goals:
            data["impedance_goals"].append({
                "z_source": [g.z_source.real, g.z_source.imag],
                "z_load": [g.z_load.real, g.z_load.imag],
                "f_min_ghz": g.f_min_ghz, "f_max_ghz": g.f_max_ghz,
                "in_port": PORT_LABEL.get(g.in_port, g.in_port),
                "out_port": PORT_LABEL.get(g.out_port, g.out_port),
                "weight": g.weight,
            })
    elif isinstance(obj, MatchingObjective):
        sgs = list(obj.goals)
    for g in sgs:
        data["sparam_goals"].append({
            "i": PORT_LABEL.get(g.i, g.i),
            "j": PORT_LABEL.get(g.j, g.j),
            "f_min_ghz": g.f_min_ghz, "f_max_ghz": g.f_max_ghz,
            "target_db": g.target_db, "weight": g.weight, "mode": g.mode,
        })
    return data

from optimizer.active_learning import (
    ActiveLearningConfig, ActiveLearningLoop, make_fdtd_fn,
)
from optimizer.ga import GAConfig, GeneticAlgorithm


# ── NL parse (fast) ────────────────────────────────────────────────────────

def parse_nl(text: str) -> tuple:
    if not text or not text.strip():
        return empty_goals_df(), "Type a description and press Parse."
    backend, model = state.llm_backend_and_model()
    try:
        objective, _ = extract_objective(text, backend=backend, model=model)
    except PromptRejectedError as e:
        return empty_goals_df(), f"⚠ Rejected: {e}"
    except Exception as e:
        return empty_goals_df(), f"❌ LLM error: {e}"
    goal_dicts = [
        {"i": g.i, "j": g.j, "f_min_ghz": g.f_min_ghz, "f_max_ghz": g.f_max_ghz,
         "target_db": g.target_db, "weight": g.weight, "mode": g.mode}
        for g in objective.goals
    ]
    label = f"({backend})" if backend else ""
    return (goals_to_df(goal_dicts),
            f"✓ Parsed {len(goal_dicts)} goals {label}: {objective.description}")


def derive_sparams_from_match(goals_df, match_df) -> tuple:
    """Append S-param return-loss + transmission goals derived from impedance rows.

    For each impedance-match goal with source Zs, load Zl in band [f_min,f_max]:
      |S_{in,in}| ≤ 20·log10(|Γ|) dB — return loss target from |Γ|=(Zl-Zs)/(Zl+Zs)
      |S_{in,out}| ≥ -1 dB          — low insertion loss in the matched band

    The rows are appended to the existing S-param goals table so the user sees
    (and can edit) the implied passive-network spec.
    """
    import pandas as pd
    igs = df_to_impedance_goals(match_df)
    if not igs:
        return goals_df, "No impedance-match rows to derive from."
    existing = goals_df if goals_df is not None else empty_goals_df()
    rows = existing.to_dict("records") if hasattr(existing, "to_dict") else []
    added = 0
    for g in igs:
        zs, zl = g.z_source, g.z_load
        gamma = abs((zl - zs) / (zl + zs)) if (zl + zs) != 0 else 0.0
        # Guardrails: bound return-loss target between -30 dB and -6 dB
        rl_db = max(-30.0, min(-6.0, 20.0 * np.log10(max(gamma, 1e-3))))
        rows.append({
            "i": PORT_LABEL[g.in_port], "j": PORT_LABEL[g.in_port],
            "f_min_ghz": g.f_min_ghz, "f_max_ghz": g.f_max_ghz,
            "target_db": round(rl_db, 1), "weight": g.weight, "mode": "below",
        })
        rows.append({
            "i": PORT_LABEL[g.in_port], "j": PORT_LABEL[g.out_port],
            "f_min_ghz": g.f_min_ghz, "f_max_ghz": g.f_max_ghz,
            "target_db": -1.0, "weight": g.weight, "mode": "above",
        })
        added += 2
    return (pd.DataFrame(rows, columns=GOAL_COLUMNS),
            f"✓ Added {added} S-param goal(s) from {len(igs)} impedance row(s).")


# ── Optimization (slow, progress + cancel) ─────────────────────────────────

def _last_summary_markdown(result: dict | None) -> str:
    if result is None:
        return ""
    goals = result["results"]["top_k"][0]["goals"]
    rows = ["| Goal | Achieved | Worst | Penalty |",
            "|------|----------|-------|---------|"]
    for g in goals:
        rows.append(
            f"| {_sparam_compass(g['goal'])} "
            f"| {g.get('achieved_db', float('nan')):+.1f} dB "
            f"| {g.get('worst_db', float('nan')):+.1f} dB "
            f"| {g['penalty']:.2f} |"
        )
    return "\n".join(rows)


def run_optimization(
    goals_df, match_df, pop_size, n_generations, seed, checkpoint,
    al_enable, al_rounds, al_candidates, al_dataset_dir, finetune_enable,
    progress=gr.Progress(track_tqdm=False),
):
    """Optimize a pixel-grid design. Emits labelled progress stages.

    Stages: Loading surrogate → Parsing goals → GA population seeding →
    GA gen k/N · best f → Local search → [FDTD Validation → AL in progress] →
    Rendering plots. Cooperatively checks `state.is_cancelled()` so the Stop
    button halts between generations / AL rounds without leaving the live
    surrogate in a half-trained state.
    """
    state.clear_cancel()
    progress(0.02, desc="Loading surrogate...")
    try:
        evaluator = state.load_evaluator(checkpoint)
    except Exception as e:
        return (None, None, None,
                f"❌ Failed to load surrogate: {e}",
                _last_summary_markdown(None))

    progress(0.04, desc="Parsing goals...")
    objective = build_objective(goals_df, match_df)
    if objective is None:
        return (None, None, None,
                "❌ No valid goals. Add at least one row.",
                _last_summary_markdown(None))

    ga_cfg = GAConfig(pop_size=int(pop_size),
                      n_generations=int(n_generations),
                      seed=int(seed))

    # The progress bar's end-point depends on whether AL follows the GA.
    ga_span = (0.05, 0.70) if al_enable else (0.05, 0.92)

    def _ga_progress(phase, gen, total, stats):
        if phase == "init":
            progress(ga_span[0], desc="GA population seeding")
        elif phase == "gen":
            frac = ga_span[0] + (ga_span[1] - ga_span[0]) * (gen / max(total, 1))
            desc = f"GA gen {gen}/{total}"
            if stats and stats.get("best_fitness") is not None:
                desc += f" · best {stats['best_fitness']:.3f}"
            progress(frac, desc=desc)
        elif phase == "local_search":
            progress(ga_span[1], desc="Local search")

    def _should_stop():
        return state.is_cancelled()

    if al_enable:
        try:
            from invdesign.simulate import VALIDATION_CONFIG
            fdtd_fn = make_fdtd_fn(
                output_dir=Path(tempfile.mkdtemp(prefix="rfdesign_al_")),
                config=VALIDATION_CONFIG,
            )
        except Exception as e:
            return (None, None, None,
                    f"❌ Could not prepare FDTD backend: {e}",
                    _last_summary_markdown(None))

        al_cfg = ActiveLearningConfig(
            n_rounds=int(al_rounds),
            candidates_per_round=int(al_candidates),
            ga_config=ga_cfg,
            dataset_dir=(Path(al_dataset_dir)
                         if al_dataset_dir and al_dataset_dir.strip()
                         else None),
        )
        loop = ActiveLearningLoop(objective, evaluator, fdtd_fn, al_cfg)
        t0 = time.time()
        n_rounds = int(al_rounds)
        for r in range(n_rounds):
            if _should_stop():
                break
            base = 0.05 + 0.80 * (r / max(n_rounds, 1))
            progress(base, desc=f"AL in progress · round {r+1}/{n_rounds} (GA)")
            try:
                loop._run_round(r, verbose=False)
            except Exception as e:
                return (None, None, None,
                        f"❌ Active learning failed: {e}",
                        _last_summary_markdown(None))
            progress(0.05 + 0.80 * ((r + 1) / max(n_rounds, 1)),
                     desc=f"FDTD Validation · round {r+1}/{n_rounds} complete "
                          f"· {len(loop.validated)} record(s)")

        if loop.config.cache_dir:
            try:
                loop._save_cache()
            except Exception:
                pass

        dt = time.time() - t0
        best_rec = max(loop.validated, key=lambda v: v.fdtd_fitness,
                        default=None)
        if best_rec is None:
            return (None, None, None,
                    "❌ No FDTD designs produced (may have been stopped early).",
                    _last_summary_markdown(None))

        pseudo_top = {
            "grid": best_rec.grid,
            "sparams": best_rec.sparams_fdtd,
            "fitness": best_rec.fdtd_fitness,
            "fill": float(best_rec.grid[1:-1, 1:-1].mean()),
            "goals": objective.evaluate(best_rec.sparams_fdtd)["goals"],
        }
        results = {
            "top_k": [pseudo_top],
            "best_fitness_history": [v.fdtd_fitness for v in loop.validated],
            "mean_fitness_history": [v.surrogate_fitness for v in loop.validated],
        }
        state.set_last_results({
            "objective": objective, "results": results,
            "grid": best_rec.grid, "sparams": best_rec.sparams_fdtd,
            "fitness": best_rec.fdtd_fitness,
        })

        progress(0.90, desc="Rendering plots")
        sfig = _fig_to_png(plot_sparams(best_rec.sparams_fdtd, objective),
                            "sparams")
        gfig = _fig_to_png(plot_grid(best_rec.grid), "grid")
        cfig = _fig_to_png(plot_convergence(
            results["best_fitness_history"],
            results["mean_fitness_history"]), "convergence")

        ft_note = ""
        if finetune_enable and loop.validated:
            grids_arr = np.stack([v.grid for v in loop.validated])
            sparams_arr = np.stack([v.sparams_fdtd for v in loop.validated])
            if state.start_finetune_background(grids_arr, sparams_arr):
                ft_note = " · fine-tune started in background"
            else:
                ft_note = " · fine-tune skipped (busy)"

        added = (f" · {len(loop.validated)} design(s) appended to "
                 f"{al_dataset_dir}" if al_cfg.dataset_dir else "")
        stopped = " (STOPPED EARLY)" if _should_stop() else ""
        status = (f"✓ AL done in {dt:.1f}s{stopped} · "
                  f"FDTD fitness {best_rec.fdtd_fitness:.2f} "
                  f"(surrogate {best_rec.surrogate_fitness:.2f}, "
                  f"gap {best_rec.gap_ratio:.2f}x){added}{ft_note}")
        progress(1.0, desc="Done")
        return (sfig, gfig, cfig, status,
                _last_summary_markdown(state.get_last_results()))

    # Plain GA path (no AL)
    ga = GeneticAlgorithm(objective, evaluator, ga_cfg)
    progress(ga_span[0],
             desc=f"GA start · pop={ga_cfg.pop_size} × gens={ga_cfg.n_generations}")
    t0 = time.time()
    try:
        results = ga.run(verbose=False, progress_callback=_ga_progress,
                          should_stop=_should_stop)
    except TypeError:
        # Older GA without should_stop/progress_callback kwargs
        results = ga.run(verbose=False)
    dt = time.time() - t0

    best = results["top_k"][0]
    state.set_last_results({
        "objective": objective, "results": results,
        "grid": best["grid"], "sparams": best["sparams"],
        "fitness": best["fitness"],
    })

    progress(0.95, desc="Rendering plots")
    sfig = _fig_to_png(plot_sparams(best["sparams"], objective), "sparams")
    gfig = _fig_to_png(plot_grid(best["grid"]), "grid")
    cfig = _fig_to_png(plot_convergence(
        results["best_fitness_history"],
        results["mean_fitness_history"]), "convergence")

    stopped = " (STOPPED EARLY)" if _should_stop() else ""
    status = (f"✓ Done in {dt:.1f}s{stopped} · "
              f"fitness {best['fitness']:.2f} · fill {best['fill']:.0%}")
    progress(1.0, desc="Done")
    return (sfig, gfig, cfig, status,
            _last_summary_markdown(state.get_last_results()))


# ── Export (fast) ──────────────────────────────────────────────────────────

def export_design(design_name: str):
    last = state.get_last_results()
    if last is None:
        return None
    from invdesign.touchstone import write_touchstone_nport

    grid = last["grid"]
    sparams = last["sparams"]
    stem = (design_name or "design").strip().replace(" ", "_") or "design"

    tmpdir = Path(tempfile.mkdtemp(prefix="rfdesign_"))
    out = tmpdir / stem
    out.mkdir()

    np.save(out / f"{stem}.npy", grid.astype(np.int8))
    try:
        from invdesign.layout_gf import write_layout_bundle
        write_layout_bundle(grid, out, stem=stem)
    except Exception:
        try:
            from invdesign.layout_klayout import write_klayout_gds
            write_klayout_gds(grid, out, stem=stem)
        except Exception:
            pass

    n_ports = sparams.shape[1]
    freq_hz = np.linspace(1.0, 30.0, sparams.shape[0]) * 1e9
    write_touchstone_nport(out / f"{stem}.s{n_ports}p",
                            n_ports, freq_hz, sparams)

    summary = [
        "RF Pixel-Grid Inverse Design Export",
        f"Design: {design_name or 'unnamed'}",
        f"Date: {datetime.now():%Y-%m-%d %H:%M}",
        f"Grid: {grid.shape[0]}×{grid.shape[1]} · fill {grid[1:-1,1:-1].mean():.1%}",
        f"Fitness: {last['fitness']:.2f}",
        "",
        "Goals:",
    ]
    for g in last["results"]["top_k"][0]["goals"]:
        summary.append(
            f"  · {_sparam_compass(g['goal'])}  "
            f"achieved {g.get('achieved_db', 0):+.1f} dB"
        )
    (out / f"{stem}_summary.txt").write_text("\n".join(summary))

    # JSON spec so the design can be round-tripped through the JSON Upload tab
    try:
        spec = _objective_to_json(last["objective"])
        spec["name"] = design_name or spec.get("name") or "design"
        (out / f"{stem}_spec.json").write_text(json.dumps(spec, indent=2))
    except Exception as e:
        print(f"[export] spec.json skipped: {e}")

    # Pixel-grid visualization PNG
    try:
        fig = plot_grid(grid)
        fig.savefig(out / f"{stem}_grid.png", dpi=160,
                    bbox_inches="tight", facecolor=fig.get_facecolor())
        import matplotlib.pyplot as plt
        plt.close(fig)
    except Exception as e:
        print(f"[export] grid.png skipped: {e}")

    zip_path = tmpdir / f"{stem}.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in out.iterdir():
            zf.write(f, f"{stem}/{f.name}")
    return str(zip_path)
