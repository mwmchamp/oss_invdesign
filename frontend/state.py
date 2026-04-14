"""Shared state for the Gradio frontend.

Holds the live surrogate evaluator, LLM backend info, last-run results, and
the background fine-tune worker. All exposed through small functions so that
other modules don't reach into this module's globals.

Design rule from past app breakage: nothing in here schedules Gradio timers
or starts SSE streams — this is pure Python state only. UI wiring lives in
`frontend.app`.
"""

from __future__ import annotations

import copy
import threading
from pathlib import Path

import numpy as np
import torch

from optimizer.ga import SurrogateEvaluator, EnsembleEvaluator
from frontend.llm_extract import (
    auto_detect_backend,
    get_backend_diagnostics,
    get_llama_load_error,
    is_llama_loaded,
    is_llama_loading,
    _load_llama,
)

# ── Core state ─────────────────────────────────────────────────────────────

_evaluator: SurrogateEvaluator | EnsembleEvaluator | None = None
_checkpoint_path: str = ""
_llm_backend: str = "fallback"
_llm_model: str = ""
_last_results: dict | None = None

# ── Cancellation ───────────────────────────────────────────────────────────
# The Stop button sets _cancel_event; long-running handlers
# (GA loop, AL loop) check `is_cancelled()` cooperatively.

_cancel_event = threading.Event()


def request_cancel() -> str:
    _cancel_event.set()
    return "⏹ Stop requested — will halt at the next safe point."


def clear_cancel() -> None:
    _cancel_event.clear()


def is_cancelled() -> bool:
    return _cancel_event.is_set()


# ── Fine-tune background worker ────────────────────────────────────────────

_finetune_lock = threading.Lock()
_finetune_busy: bool = False
_finetune_status: str = ""
_finetune_last_ok: str = ""


def finetune_busy() -> bool:
    with _finetune_lock:
        return _finetune_busy


def get_finetune_status() -> str:
    with _finetune_lock:
        if _finetune_busy:
            return f"🟡 {_finetune_status}"
        if _finetune_last_ok:
            return f"🟢 {_finetune_last_ok}"
    return ""


def _members_of(ev) -> list[SurrogateEvaluator]:
    if isinstance(ev, EnsembleEvaluator):
        return ev.members
    return [ev]


def _finetune_member_copy(member, grids, targets_norm,
                          n_steps: int = 40, lr: float = 1e-4) -> dict:
    """Fine-tune a deep copy; return its state_dict. No mutation of ``member``."""
    model_copy = copy.deepcopy(member.model)
    model_copy.train()
    opt = torch.optim.Adam(model_copy.parameters(), lr=lr)
    device = member.device
    x = torch.from_numpy(grids.astype(np.float32)).unsqueeze(1).to(device)
    y = torch.from_numpy(targets_norm.astype(np.float32)).to(device)

    final_loss = float("nan")
    for _ in range(n_steps):
        pred = model_copy(x)
        loss = torch.nn.functional.mse_loss(pred, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        final_loss = float(loss.item())
        if not np.isfinite(final_loss):
            raise RuntimeError(f"non-finite loss: {final_loss}")

    model_copy.eval()
    return {"state_dict": model_copy.state_dict(), "final_loss": final_loss}


def start_finetune_background(grids: np.ndarray, sparams: np.ndarray,
                              n_steps: int = 40, lr: float = 1e-4) -> bool:
    """Start a fine-tune in the background on validated AL records.

    The live model weights are only overwritten *after* every member finishes
    training cleanly. Any exception aborts the swap; the live model stays on
    the pre-fine-tune weights. Returns False if no evaluator or already busy.
    """
    global _finetune_busy, _finetune_status
    from surrogate.data import sparams_to_upper_tri

    with _finetune_lock:
        if _finetune_busy or _evaluator is None:
            return False
        if grids is None or len(grids) == 0:
            return False
        _finetune_busy = True
        _finetune_status = f"Fine-tuning on {len(grids)} FDTD record(s)..."

    def _worker():
        global _finetune_busy, _finetune_status, _finetune_last_ok
        try:
            members = _members_of(_evaluator)
            targets_raw = np.stack([sparams_to_upper_tri(s) for s in sparams])
            new_states = []
            for i, m in enumerate(members):
                with _finetune_lock:
                    _finetune_status = (
                        f"Fine-tuning member {i+1}/{len(members)} "
                        f"on {len(grids)} FDTD record(s)..."
                    )
                mean = m.target_mean.cpu().numpy()
                std = m.target_std.cpu().numpy()
                targets_norm = (targets_raw - mean) / np.maximum(std, 1e-8)
                res = _finetune_member_copy(
                    m, grids, targets_norm, n_steps=n_steps, lr=lr,
                )
                new_states.append(res)

            # Atomic swap only after every member succeeded.
            for m, res in zip(members, new_states):
                m.model.load_state_dict(res["state_dict"])
                m.model.eval()

            losses = [f"{r['final_loss']:.4g}" for r in new_states]
            with _finetune_lock:
                _finetune_last_ok = (
                    f"Fine-tune ✓ (members: loss→{', '.join(losses)})"
                )
                _finetune_status = ""
        except Exception as e:
            with _finetune_lock:
                _finetune_status = f"Fine-tune failed (weights unchanged): {e}"
        finally:
            with _finetune_lock:
                _finetune_busy = False

    threading.Thread(target=_worker, daemon=True).start()
    return True


# ── Evaluator loading (lazy) ───────────────────────────────────────────────

def load_evaluator(checkpoint: str, device: str | None = None):
    """Load (or reuse) the evaluator. Supports single ckpt, comma list, or dir."""
    global _evaluator, _checkpoint_path
    if _evaluator is not None and _checkpoint_path == checkpoint:
        return _evaluator
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    paths = [p.strip() for p in checkpoint.split(",") if p.strip()]
    if len(paths) == 1:
        p = Path(paths[0])
        if p.is_dir():
            members = sorted(p.glob("member_*/best_model.pt"))
            if members:
                paths = [str(m) for m in members]
    if len(paths) > 1:
        _evaluator = EnsembleEvaluator(paths, device=device)
    else:
        _evaluator = SurrogateEvaluator(paths[0], device=device)
    _checkpoint_path = checkpoint
    return _evaluator


def current_evaluator():
    return _evaluator


# ── Last-results accessor (for Export) ─────────────────────────────────────

def set_last_results(r: dict | None) -> None:
    global _last_results
    _last_results = r


def get_last_results() -> dict | None:
    return _last_results


# ── LLM status ─────────────────────────────────────────────────────────────

def preload_llm_in_background() -> None:
    def _worker():
        global _llm_backend, _llm_model
        backend, model, _, _ = auto_detect_backend()
        _llm_backend, _llm_model = backend, model
        if backend == "llama":
            try:
                _load_llama(model)
            except Exception as e:
                print(f"[llm] preload failed: {e}")
    threading.Thread(target=_worker, daemon=True).start()


def llm_backend_and_model() -> tuple[str, str]:
    return _llm_backend, _llm_model


def refresh_llm_status() -> str:
    """Re-detect backend (picks up Ollama started after app launch) and return status."""
    global _llm_backend, _llm_model
    if not is_llama_loaded() and not is_llama_loading():
        backend, model, _, _ = auto_detect_backend()
        _llm_backend, _llm_model = backend, model
        if backend == "llama":
            # Kick off load in background now that we've (re)detected it
            preload_llm_in_background()
    return get_llm_status()


def get_llm_status() -> str:
    diag = get_backend_diagnostics()
    diag_tail = ("<br><span style='opacity:0.7;font-size:0.85em'>"
                 + " · ".join(diag) + "</span>") if diag else ""
    if _llm_backend == "llama":
        if is_llama_loaded():
            return f"🟢 Llama 3.2 loaded{diag_tail}"
        if is_llama_loading():
            return f"🟡 Loading Llama...{diag_tail}"
        err = get_llama_load_error()
        if err:
            return f"🔴 Llama load failed: {err}{diag_tail}"
        return f"⚪ Llama pending load{diag_tail}"
    if _llm_backend == "ollama":
        return f"🟢 Ollama ({_llm_model}){diag_tail}"
    if _llm_backend == "fallback":
        return f"⚪ Regex fallback (no LLM){diag_tail}"
    return "—"
