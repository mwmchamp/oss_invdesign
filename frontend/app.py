"""Gradio frontend for RF pixel-grid inverse design.

Thin UI wiring layer. Heavy lifting lives in sibling modules:
  - `state`    : shared process state (evaluator, LLM, fine-tune, cancel)
  - `goals_io` : DataFrame / JSON schema conversion, mode legend
  - `plots`    : matplotlib figures for the results panel
  - `presets`  : canonical one-click starting points
  - `ops`      : slow-path click handlers (parse, optimize, export)

Usage:
    python -m frontend.app [--checkpoint PATH] [--port PORT] [--share]
"""

from __future__ import annotations

import argparse
import os

import gradio as gr

from frontend import state, ops
from frontend.goals_io import (
    GOAL_COLUMNS, MATCH_COLUMNS, MODE_LEGEND_MD,
    empty_goals_df, empty_match_df, parse_specs_json,
)
from frontend.presets import PRESETS, PRESET_NAMES, apply_preset


# Latest model: disconnect-augmentation fine-tune (run_v4_karahan_discaug_v2).
# Drops the hallucination-probe outputs (empty / disjoint grids) from
# ~-28 dB to <-38 dB vs. the DRAFT run_v4_karahan checkpoint, at the cost
# of ~0.02 dB on aggregate magnitude error. Report Table 4.1 / Fig 7.1.
DEFAULT_CHECKPOINT = os.environ.get(
    "INVDESIGN_CKPT", "./checkpoints/best_model.pt"
)
DEFAULT_DATASET_DIR = os.environ.get(
    "INVDESIGN_DATASET", "./datasets/pixelgrid"
)


_CSS = """
#main { max-width: 1400px; margin: 0 auto; }
.preset-row .gr-button { min-height: 44px; font-size: 0.95em; }
.status-box { font-family: ui-monospace, Menlo, monospace; }
.legend-md { font-size: 0.9em; opacity: 0.85; margin-top: -4px; }
"""


def build_app(default_checkpoint: str) -> gr.Blocks:
    # Gradio 6: theme/css belong to launch(); Blocks() takes only `title`.
    with gr.Blocks(title="RF Pixel-Grid Inverse Design") as demo:
        gr.Markdown(
            "## RF Pixel-Grid Inverse Design\n"
            "Describe a passive structure or upload a spec; the surrogate + "
            "GA returns a 27×27 pixel layout. Sweet spot is 10–15 GHz on "
            "IHP SG13G2."
        )

        with gr.Row(elem_id="main"):
            # ── Left: input ────────────────────────────────────────────
            with gr.Column(scale=5):
                with gr.Tabs():
                    # --- Tab 1: Natural Language ---------------------
                    with gr.Tab("Natural Language"):
                        with gr.Row(elem_classes="preset-row"):
                            preset_btns = [
                                gr.Button(name, variant="secondary", size="sm")
                                for name in PRESET_NAMES
                            ]
                        nl_input = gr.Textbox(
                            label="Describe the design",
                            placeholder="e.g. '15 GHz high-pass filter: pass "
                                        "above 15 GHz, reject below 10 GHz.'",
                            lines=3,
                        )
                        with gr.Row():
                            parse_btn = gr.Button("Parse →", variant="primary")
                            llm_status = gr.Markdown(state.get_llm_status())
                            refresh_llm_btn = gr.Button("↻", size="sm", scale=0)
                        nl_status = gr.Markdown("")

                    # --- Tab 2: JSON Upload --------------------------
                    with gr.Tab("JSON Upload"):
                        gr.Markdown(
                            "Upload a specs JSON to bypass the LLM. Schema: "
                            "`{name, sparam_goals: [...], "
                            "impedance_goals: [...]}`."
                        )
                        specs_file = gr.File(
                            label="Specs JSON", file_types=[".json"],
                            file_count="single",
                        )
                        load_json_btn = gr.Button("Load JSON →",
                                                   variant="primary")
                        json_status = gr.Markdown("")

                # ── Parameters section (shared across both tabs) ──────
                gr.Markdown("### Parameters")
                gr.Markdown(MODE_LEGEND_MD, elem_classes="legend-md")
                goals_df = gr.Dataframe(
                    value=empty_goals_df(),
                    headers=GOAL_COLUMNS,
                    datatype=["str", "str", "number", "number",
                              "number", "number", "str"],
                    interactive=True,
                    label="S-parameter goals",
                    wrap=True,
                )
                with gr.Accordion("Impedance match goals (optional)",
                                   open=False):
                    match_df = gr.Dataframe(
                        value=empty_match_df(),
                        headers=MATCH_COLUMNS,
                        datatype=["number"] * 6 + ["str", "str", "number"],
                        interactive=True,
                        label="Complex impedance matching",
                        wrap=True,
                    )
                    with gr.Row():
                        derive_sp_btn = gr.Button(
                            "Find S-params from match →", size="sm",
                        )
                    derive_sp_status = gr.Markdown("")

                # ── GA settings ───────────────────────────────────────
                with gr.Accordion("GA settings", open=False):
                    checkpoint_box = gr.Textbox(
                        label="Surrogate checkpoint / ensemble dir",
                        value=default_checkpoint, lines=1,
                    )
                    with gr.Row():
                        pop_size = gr.Number(label="Population", value=120,
                                              precision=0)
                        n_generations = gr.Number(label="Generations",
                                                    value=40, precision=0)
                        seed = gr.Number(label="Seed", value=42, precision=0)

                # ── Active learning ────────────────────────────────────
                with gr.Accordion("Active learning (FDTD in the loop)",
                                   open=False):
                    al_enable = gr.Checkbox(
                        label="Enable active learning", value=False,
                    )
                    with gr.Row():
                        al_rounds = gr.Number(label="Rounds", value=2,
                                               precision=0)
                        al_candidates = gr.Number(label="Candidates/round",
                                                    value=2, precision=0)
                    al_dataset_dir = gr.Textbox(
                        label="Append validated designs to dataset dir "
                              "(optional)",
                        value=DEFAULT_DATASET_DIR, lines=1,
                    )
                    finetune_enable = gr.Checkbox(
                        label="Fine-tune surrogate in background on "
                              "validated designs",
                        value=True,
                    )
                    ft_status = gr.Markdown("")
                    refresh_ft_btn = gr.Button("↻ Refresh fine-tune status",
                                                 size="sm")

                # ── Run / Stop ────────────────────────────────────────
                with gr.Row():
                    run_btn = gr.Button("Optimize", variant="primary",
                                          scale=3)
                    stop_btn = gr.Button("⏹ Stop", variant="stop", scale=1)
                run_status = gr.Markdown("")

            # ── Right: results ────────────────────────────────────────
            with gr.Column(scale=7):
                # gr.Image (not gr.Plot) so every output has a built-in
                # download button and a click-to-expand lightbox.
                # gr.Image (not gr.Plot) so every output has a built-in
                # download button and a click-to-expand fullscreen view.
                _img_kw = dict(type="filepath", interactive=False,
                               buttons=["download", "fullscreen"])
                sparam_plot = gr.Image(label="S-parameters", **_img_kw)
                with gr.Row():
                    grid_plot = gr.Image(label="Best design", **_img_kw)
                    conv_plot = gr.Image(label="Convergence", **_img_kw)
                summary_md = gr.Markdown("")
                with gr.Accordion("Export", open=False):
                    design_name = gr.Textbox(label="Design name",
                                              value="design", lines=1)
                    export_btn = gr.Button("Build export ZIP")
                    export_file = gr.File(label="Download", interactive=False)

        # ── Wiring ────────────────────────────────────────────────────

        parse_btn.click(
            fn=ops.parse_nl, inputs=nl_input,
            outputs=[goals_df, nl_status],
        )

        # Preset buttons: pass the preset name via gr.State (more robust
        # than capturing in a lambda closure, which can bind oddly in
        # Gradio 6's event graph).
        for btn, name in zip(preset_btns, PRESET_NAMES):
            preset_name_state = gr.State(name)
            btn.click(
                fn=apply_preset,
                inputs=preset_name_state,
                outputs=[nl_input, goals_df, match_df],
            )

        load_json_btn.click(
            fn=parse_specs_json, inputs=specs_file,
            outputs=[goals_df, match_df, json_status],
        )

        derive_sp_btn.click(
            fn=ops.derive_sparams_from_match,
            inputs=[goals_df, match_df],
            outputs=[goals_df, derive_sp_status],
        )

        refresh_llm_btn.click(fn=state.refresh_llm_status,
                               inputs=[], outputs=llm_status)
        refresh_ft_btn.click(fn=state.get_finetune_status,
                              inputs=[], outputs=ft_status)

        # NOTE: no `cancels=[run_evt]`. In Gradio 6, pairing `cancels=`
        # with `concurrency_limit=1` on the cancelled event can deadlock
        # the queue — subsequent clicks on ANY button silently drop.
        # Instead, Stop flips a threading.Event that the GA/AL loops
        # check cooperatively.
        run_btn.click(
            fn=ops.run_optimization,
            inputs=[goals_df, match_df, pop_size, n_generations, seed,
                    checkpoint_box, al_enable, al_rounds, al_candidates,
                    al_dataset_dir, finetune_enable],
            outputs=[sparam_plot, grid_plot, conv_plot, run_status,
                     summary_md],
            concurrency_id="optimize", concurrency_limit=1,
        )

        stop_btn.click(fn=state.request_cancel,
                       inputs=[], outputs=run_status)

        export_btn.click(fn=ops.export_design, inputs=design_name,
                         outputs=export_file)

    return demo


def main():
    parser = argparse.ArgumentParser(
        description="RF pixel-grid inverse-design GUI",
    )
    parser.add_argument(
        "--checkpoint", default=DEFAULT_CHECKPOINT,
        help="Surrogate checkpoint file or ensemble directory.",
    )
    parser.add_argument("--port", type=int, default=None,
                        help="Listen port (default: Gradio picks from 7860+).")
    parser.add_argument("--share", action="store_true",
                        help="Create a public Gradio share link.")
    args = parser.parse_args()

    state.preload_llm_in_background()
    demo = build_app(args.checkpoint)
    demo.queue(default_concurrency_limit=5, max_size=32)
    demo.launch(
        server_port=args.port,
        share=args.share,
        show_error=True,
        theme=gr.themes.Soft(primary_hue="indigo", secondary_hue="slate"),
        css=_CSS,
    )


if __name__ == "__main__":
    main()
