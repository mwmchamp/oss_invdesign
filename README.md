# Survival of the Fastest: Rapid 4-Port Passive RFIC Optimization via Surrogate-Assisted Genetic Algorithms

Minimal, portable release of the RF pixel-grid inverse design pipeline from the thesis work.

## Included modules
- `invdesign/`: dataset generation, OpenEMS runner, GDS export
- `surrogate/`: model definition, data loader, training entrypoint
- `optimizer/`: GA, objectives, active learning, validation, hi-fi OpenEMS wrapper
- `frontend/`: Gradio app
- `tests/test_ga_vs_simulator.py`: regression test
- `scripts/`: representative scripts for dataset generation, training, hi-fi eval, and preset gallery generation

## Quick start
1. Create an environment with the Python dependencies used by your workflow (`numpy`, `torch`, `gradio`, and OpenEMS bindings/libraries for EM simulation).
2. Place or download a surrogate checkpoint (for example `./checkpoints/best_model.pt`).
3. Launch the app:

```bash
./launch.sh --port 7860
```

## Environment variables
- `INVDESIGN_CKPT`: surrogate checkpoint path (default `./checkpoints/best_model.pt`)
- `INVDESIGN_DATASET`: dataset directory path
- `INVDESIGN_LLAMA_DIR`: local Llama model directory for natural-language preset parsing
- `OPENEMS_LIB`: directory containing OpenEMS shared libraries (prepended to `LD_LIBRARY_PATH`)

## Representative scripts
- Dataset generation: `scripts/generate_dataset.sh`
- Surrogate training: `scripts/train_surrogate.sh`
- High-fidelity eval: `scripts/run_hifi_eval.sh`
- Preset gallery build: `python3 scripts/build_preset_gallery.py`

Each script documents its required environment variables near the top.
