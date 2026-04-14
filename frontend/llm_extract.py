"""Extract matching network parameters from natural language using an LLM.

Supports multiple backends:
  - "llama": local Llama 3.2 (Meta checkpoint format, tiktoken tokenizer)
  - "ollama": local Ollama server (requires `ollama serve`)
  - "fallback": regex-based extraction, no LLM needed

The LLM receives a structured prompt and returns JSON with S-parameter goals,
or a rejection if the request is not an RF/microwave design task.
"""

from __future__ import annotations

import json
import os
import re

from optimizer.objectives import SParamGoal, MatchingObjective

# Default Llama 3.2 3B Instruct directory. Override via $INVDESIGN_LLAMA_DIR.
DEFAULT_LLAMA_DIR = os.environ.get(
    "INVDESIGN_LLAMA_DIR", "./models/Llama-3.2-3B-Instruct"
)

# System prompt that instructs the LLM to extract RF parameters
SYSTEM_PROMPT = """You translate natural-language RF design requests into a JSON S-parameter specification for a 27x27 pixel-grid inverse-design tool. The tool targets IHP SG13G2 130nm SiGe BiCMOS passive structures (filters, couplers, dividers, matching networks). The sweet-spot frequency band is 10-15 GHz; full supported range is 1-30 GHz.

# OUTPUT FORMAT
Return ONLY a single JSON object. No prose, no markdown, no code fences.

Reject irrelevant requests with:
{"rejected": true, "reason": "<one sentence>"}

Accept RF-design requests with:
{
  "name": "short_snake_case_id",
  "description": "<one-line human summary>",
  "goals": [ <goal-object>, ... ]
}

Each goal-object has EXACTLY these keys:
  "i":           int 0-3 (source port)
  "j":           int 0-3 (destination port; i==j means reflection/return-loss)
  "f_min_ghz":   number (0.1 to 30.0)
  "f_max_ghz":   number (> f_min_ghz, <= 30.0)
  "target_db":   number (-100 to +5, the |S_ij| target in dB)
  "weight":      number (0.5 to 10.0, importance)
  "mode":        "above" | "below" | "at"

# PORT CONVENTION (FIXED)
Port 0 = N (North edge)   Port 1 = S (South edge)
Port 2 = E (East edge)    Port 3 = W (West edge)
The grid is symmetric, so any physical port label maps to one of N/S/E/W.

# MODE SEMANTICS
"above" = we want |S_ij| >= target (transmission / passband goals).
"below" = we want |S_ij| <= target (rejection / return-loss / isolation).
"at"    = we want |S_ij| ~= target (equal-split couplers where the exact value matters).

# S-PARAMETER MEANING
|S_ij| is the power transferred from port j into port i (i!=j: transmission/coupling; i==j: reflection).
S11 low (e.g. -10 dB) means port 0 is well matched. S01 high means port 0 and port 1 are strongly coupled.

# COMPOSITION RULES (apply to every response)
1. For any 2-port design (signal flowing between two ports, e.g. N<->S), ALWAYS include:
     - the transmission goal on S(out,in) "above" in the passband,
     - a return-loss goal S(in,in) "below" -10 dB in the passband,
     - isolation goals on the two unused ports: S(unused, in) "below" -15 to -20 dB, weight 0.5, across the full 1-30 GHz band.
2. For 4-port designs (couplers, dividers, crossovers), specify goals on every signal port; do NOT add blanket isolation to signal ports.
3. Frequencies must lie in [1, 30] GHz; if the user specifies a band outside this, clip to 1-30 and note it in "description".
4. Weight convention: primary goal 5.0, secondary (return loss / stopband) 2-3, isolation 0.5-1.
5. If the request is ambiguous, pick reasonable defaults from the center of the user's band, prefer 10-15 GHz.
6. Impedance-matching requests phrased in complex ohms (e.g. "match to 25+j30 ohms") are ALLOWED: represent them with S11 "below" -10 dB in-band plus S21 "above" -3 dB in-band; note the load impedance in "description" so the user can open the Advanced tab for exact Zl handling.

# FEW-SHOT EXAMPLES

User: "12 GHz bandpass filter with 2 GHz bandwidth, at least 15 dB out-of-band rejection"
Output:
{"name":"bandpass_12ghz","description":"12 GHz bandpass, 11-13 GHz passband, >15 dB rejection","goals":[
 {"i":0,"j":1,"f_min_ghz":11.0,"f_max_ghz":13.0,"target_db":-3.0,"weight":5.0,"mode":"above"},
 {"i":0,"j":0,"f_min_ghz":11.0,"f_max_ghz":13.0,"target_db":-10.0,"weight":3.0,"mode":"below"},
 {"i":0,"j":1,"f_min_ghz":1.0,"f_max_ghz":9.0,"target_db":-15.0,"weight":2.0,"mode":"below"},
 {"i":0,"j":1,"f_min_ghz":15.0,"f_max_ghz":30.0,"target_db":-15.0,"weight":2.0,"mode":"below"},
 {"i":0,"j":2,"f_min_ghz":1.0,"f_max_ghz":30.0,"target_db":-20.0,"weight":0.5,"mode":"below"},
 {"i":0,"j":3,"f_min_ghz":1.0,"f_max_ghz":30.0,"target_db":-20.0,"weight":0.5,"mode":"below"}]}

User: "broadband 50-ohm through path from 10 to 15 GHz"
Output:
{"name":"broadband_10_15ghz","description":"Broadband through N->S, 10-15 GHz","goals":[
 {"i":0,"j":1,"f_min_ghz":10.0,"f_max_ghz":15.0,"target_db":-3.0,"weight":5.0,"mode":"above"},
 {"i":0,"j":0,"f_min_ghz":10.0,"f_max_ghz":15.0,"target_db":-10.0,"weight":3.0,"mode":"below"},
 {"i":0,"j":2,"f_min_ghz":1.0,"f_max_ghz":30.0,"target_db":-20.0,"weight":0.5,"mode":"below"},
 {"i":0,"j":3,"f_min_ghz":1.0,"f_max_ghz":30.0,"target_db":-20.0,"weight":0.5,"mode":"below"}]}

User: "12 GHz 6-dB coupler with high directivity"
Output:
{"name":"coupler_6db_12ghz","description":"12 GHz directional coupler, -6 dB coupling, >25 dB directivity","goals":[
 {"i":0,"j":1,"f_min_ghz":10.0,"f_max_ghz":14.0,"target_db":-1.5,"weight":3.0,"mode":"above"},
 {"i":0,"j":2,"f_min_ghz":10.0,"f_max_ghz":14.0,"target_db":-6.0,"weight":5.0,"mode":"at"},
 {"i":0,"j":3,"f_min_ghz":10.0,"f_max_ghz":14.0,"target_db":-25.0,"weight":3.0,"mode":"below"},
 {"i":0,"j":0,"f_min_ghz":10.0,"f_max_ghz":14.0,"target_db":-15.0,"weight":2.0,"mode":"below"}]}

User: "match 50 ohm source to 25+j25 ohm load at 12 GHz"
Output:
{"name":"zmatch_25p25j_12ghz","description":"Match Zs=50 to Zl=25+j25 ohm at 12 GHz (use Advanced tab for exact Zl)","goals":[
 {"i":0,"j":1,"f_min_ghz":11.0,"f_max_ghz":13.0,"target_db":-3.0,"weight":5.0,"mode":"above"},
 {"i":0,"j":0,"f_min_ghz":11.0,"f_max_ghz":13.0,"target_db":-10.0,"weight":5.0,"mode":"below"},
 {"i":0,"j":2,"f_min_ghz":1.0,"f_max_ghz":30.0,"target_db":-20.0,"weight":0.5,"mode":"below"},
 {"i":0,"j":3,"f_min_ghz":1.0,"f_max_ghz":30.0,"target_db":-20.0,"weight":0.5,"mode":"below"}]}

User: "what's the weather tomorrow"
Output:
{"rejected":true,"reason":"Not an RF design request."}

Return ONLY the JSON object. No commentary."""


# ---------------------------------------------------------------------------
# Llama 3.2 local inference (Meta checkpoint + tiktoken)
# ---------------------------------------------------------------------------

# Global cache — loaded once, reused across calls
_llama_model = None
_llama_tokenizer = None
_llama_loading = False  # True while load is in progress
_llama_device = None  # Device model is loaded on
_llama_load_error: str | None = None  # Last load failure reason (for UI)


def get_llama_load_error() -> str | None:
    return _llama_load_error


def is_llama_loaded() -> bool:
    """Check if Llama model is loaded and ready."""
    return _llama_model is not None


def is_llama_loading() -> bool:
    """Check if Llama model is currently being loaded."""
    return _llama_loading


def get_llama_device() -> str | None:
    """Return device string the model is on, or None if not loaded."""
    return _llama_device


def _load_llama(checkpoint_dir: str, device: str | None = None,
                progress_callback=None):
    """Load Llama 3.2 model and tiktoken tokenizer from Meta checkpoint.

    Args:
        checkpoint_dir: Path to Meta-format checkpoint directory.
        device: Force device ("cpu" or "cuda"). Auto-detects if None.
        progress_callback: Optional callable(step: str, pct: float) for progress.
    """
    global _llama_model, _llama_tokenizer, _llama_loading, _llama_device
    global _llama_load_error
    if _llama_model is not None:
        return

    _llama_loading = True
    _llama_load_error = None
    try:
        import torch
        import tiktoken
        import os

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        if progress_callback:
            progress_callback("Loading tokenizer...", 0.1)

        # --- tokenizer ---
        tok_path = os.path.join(checkpoint_dir, "tokenizer.model")
        mergeable_ranks = {}
        import base64
        with open(tok_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 2:
                    mergeable_ranks[base64.b64decode(parts[0])] = int(parts[1])

        special_tokens = {
            "<|begin_of_text|>": 128000,
            "<|end_of_text|>": 128001,
            "<|start_header_id|>": 128006,
            "<|end_header_id|>": 128007,
            "<|eom_id|>": 128008,
            "<|eot_id|>": 128009,
            "<|python_tag|>": 128010,
        }
        for i in range(256):
            special_tokens[f"<|reserved_special_token_{i}|>"] = 128011 + i

        _llama_tokenizer = tiktoken.Encoding(
            name="llama3",
            pat_str=r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+",
            mergeable_ranks=mergeable_ranks,
            special_tokens=special_tokens,
        )

        if progress_callback:
            progress_callback("Reading model config...", 0.2)

        # --- model ---
        with open(os.path.join(checkpoint_dir, "params.json")) as f:
            params = json.load(f)

        from frontend._llama_model import Transformer, ModelArgs
        args = ModelArgs(
            dim=params["dim"],
            n_layers=params["n_layers"],
            n_heads=params["n_heads"],
            n_kv_heads=params.get("n_kv_heads", params["n_heads"]),
            vocab_size=params["vocab_size"],
            multiple_of=params.get("multiple_of", 256),
            ffn_dim_multiplier=params.get("ffn_dim_multiplier", None),
            norm_eps=params.get("norm_eps", 1e-5),
            rope_theta=params.get("rope_theta", 500000.0),
            max_seq_len=2048,
        )

        if progress_callback:
            progress_callback("Loading model weights (this may take a moment)...", 0.4)

        checkpoint = torch.load(
            os.path.join(checkpoint_dir, "consolidated.00.pth"),
            map_location=device,
            weights_only=True,
        )

        if progress_callback:
            progress_callback("Initializing model...", 0.7)

        model = Transformer(args)
        model.load_state_dict(checkpoint, strict=False)

        if progress_callback:
            progress_callback(f"Moving to {device}...", 0.85)

        model = model.to(device=device, dtype=torch.bfloat16)
        model.eval()
        _llama_model = model
        _llama_device = device

        if progress_callback:
            progress_callback("Ready!", 1.0)

        print(f"Loaded Llama 3.2 ({params['n_layers']}L, {params['dim']}d) on {device}")
    except Exception as e:
        _llama_load_error = f"{type(e).__name__}: {e}"
        print(f"[llama] load failed: {_llama_load_error}")
        raise
    finally:
        _llama_loading = False


def _call_llama(prompt: str, checkpoint_dir: str = DEFAULT_LLAMA_DIR) -> str:
    """Run Llama 3.2 inference on a single prompt with KV-cache."""
    import torch

    _load_llama(checkpoint_dir)
    model = _llama_model
    enc = _llama_tokenizer

    # Format as Llama 3 chat
    BOS = enc.encode("<|begin_of_text|>", allowed_special="all")
    tokens = BOS
    for role, content in [("system", SYSTEM_PROMPT), ("user", prompt)]:
        tokens += enc.encode(f"<|start_header_id|>{role}<|end_header_id|>\n\n", allowed_special="all")
        tokens += enc.encode(content)
        tokens += enc.encode("<|eot_id|>", allowed_special="all")
    # Start assistant turn
    tokens += enc.encode("<|start_header_id|>assistant<|end_header_id|>\n\n", allowed_special="all")

    device = next(model.parameters()).device
    input_ids = torch.tensor([tokens], dtype=torch.long, device=device)
    eot_id = enc.encode("<|eot_id|>", allowed_special="all")[0]
    generated = []

    model.reset_cache()
    with torch.no_grad():
        # Prefill: process entire prompt at once, cache KV
        logits = model(input_ids, start_pos=0, use_cache=True)
        logits = logits[:, -1, :] / 0.1
        next_token = torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1)
        generated.append(next_token.item())
        pos = input_ids.shape[1]

        # Decode: one token at a time using cached KV
        for _ in range(511):
            if generated[-1] == eot_id:
                generated.pop()  # don't include stop token
                break
            logits = model(next_token, start_pos=pos, use_cache=True)
            logits = logits[:, -1, :] / 0.1
            next_token = torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1)
            generated.append(next_token.item())
            pos += 1

    model.reset_cache()  # free memory
    return enc.decode(generated)


def _call_ollama(prompt: str, model: str = "llama3.2", base_url: str = "http://localhost:11434") -> str:
    """Call local Ollama server."""
    import urllib.request
    req = urllib.request.Request(
        f"{base_url}/api/generate",
        data=json.dumps({
            "model": model,
            "prompt": prompt,
            "system": SYSTEM_PROMPT,
            "stream": False,
            "options": {"temperature": 0.1},
        }).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read())["response"]


class PromptRejectedError(Exception):
    """Raised when the LLM determines the prompt is not an RF design task."""
    pass


def has_gpu() -> bool:
    """Check if a CUDA GPU is available."""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        import os
        return os.path.exists("/dev/nvidiactl") or os.environ.get("CUDA_VISIBLE_DEVICES", "") != ""


_backend_diagnostics: list[str] = []


def get_backend_diagnostics() -> list[str]:
    """Return why each backend was accepted or skipped on last detection."""
    return list(_backend_diagnostics)


def auto_detect_backend() -> tuple[str, str, str, str]:
    """Auto-detect best available LLM backend from environment.

    Returns (backend, model, base_url, api_key).
    Priority: Llama (GPU only) > Ollama > regex fallback.
    Llama 3B is too slow on CPU (~minutes/token), so only use it when GPU is available.
    The regex fallback handles all 10 component types and is instant.
    """
    import os
    _backend_diagnostics.clear()

    # Check for local Llama checkpoint — only if GPU available (3B is too slow on CPU)
    gpu = has_gpu()
    llama_dir = os.environ.get("LLAMA_CHECKPOINT_DIR", DEFAULT_LLAMA_DIR)
    ckpt_path = os.path.join(llama_dir, "consolidated.00.pth")
    if gpu and os.path.isfile(ckpt_path):
        _backend_diagnostics.append(f"Llama: GPU + checkpoint at {llama_dir}")
        return "llama", llama_dir, "", ""
    if not gpu:
        _backend_diagnostics.append("Llama skipped: no CUDA GPU (3B too slow on CPU)")
    elif not os.path.isfile(ckpt_path):
        _backend_diagnostics.append(f"Llama skipped: checkpoint not found at {ckpt_path}")

    # Check local Ollama
    try:
        import urllib.request
        urllib.request.urlopen("http://localhost:11434/api/tags", timeout=2)
        _backend_diagnostics.append("Ollama: localhost:11434 reachable")
        return "ollama", "llama3.2", "http://localhost:11434", ""
    except Exception as e:
        _backend_diagnostics.append(f"Ollama skipped: {type(e).__name__} on localhost:11434")

    _backend_diagnostics.append("Using regex fallback (no LLM available)")
    return "fallback", "", "", ""


def _fallback_extract(text: str) -> dict:
    """Regex-based parameter extraction when no LLM is available.

    Handles common patterns like:
      "5 GHz bandpass filter"
      "lowpass with 10 GHz cutoff"
      "broadband match from 1 to 20 GHz"
      "notch at 10 GHz"
      "bandpass 2-6 GHz with -15dB rejection"
    """
    text_lower = text.lower()
    goals = []

    # Extract frequencies mentioned
    # First try "X to Y GHz" / "X-Y GHz" patterns (number without unit followed by number with unit)
    range_pattern = r'(\d+(?:\.\d+)?)\s*(?:to|-)\s*(\d+(?:\.\d+)?)\s*(?:ghz|GHz)'
    range_match = re.search(range_pattern, text)
    if range_match:
        freqs = [float(range_match.group(1)), float(range_match.group(2))]
    else:
        freq_pattern = r'(\d+(?:\.\d+)?)\s*(?:ghz|GHz)'
        freqs = [float(m) for m in re.findall(freq_pattern, text)]

    # Extract dB targets
    db_pattern = r'(-?\d+(?:\.\d+)?)\s*(?:db|dB)'
    dbs = [float(m) for m in re.findall(db_pattern, text)]

    # Detect filter type
    if any(w in text_lower for w in ["bandpass", "band-pass", "band pass", "bpf"]):
        # Bandpass filter
        if len(freqs) >= 2:
            f_lo, f_hi = sorted(freqs[:2])
        elif len(freqs) == 1:
            f_lo, f_hi = freqs[0] * 0.9, freqs[0] * 1.1
        else:
            f_lo, f_hi = 4.5, 5.5

        rejection_db = dbs[0] if dbs else -15.0
        goals = [
            {"i": 0, "j": 1, "f_min_ghz": f_lo, "f_max_ghz": f_hi,
             "target_db": -3.0, "weight": 5.0, "mode": "above"},
            {"i": 0, "j": 0, "f_min_ghz": f_lo, "f_max_ghz": f_hi,
             "target_db": -10.0, "weight": 3.0, "mode": "below"},
            {"i": 0, "j": 1, "f_min_ghz": 1.0, "f_max_ghz": max(1.0, f_lo - 1.0),
             "target_db": rejection_db, "weight": 1.0, "mode": "below"},
            {"i": 0, "j": 1, "f_min_ghz": f_hi + 2.0, "f_max_ghz": min(30.0, f_hi + 10.0),
             "target_db": rejection_db, "weight": 1.0, "mode": "below"},
        ]
        name = f"bandpass_{f_lo:.0f}_{f_hi:.0f}ghz"
        desc = f"Bandpass {f_lo:.1f}-{f_hi:.1f} GHz"

    elif any(w in text_lower for w in ["notch", "band-stop", "bandstop", "reject"]):
        # Notch filter
        if len(freqs) >= 2:
            f_lo, f_hi = sorted(freqs[:2])
        elif len(freqs) == 1:
            f_lo, f_hi = freqs[0] - 1.0, freqs[0] + 1.0
        else:
            f_lo, f_hi = 9.0, 11.0

        goals = [
            {"i": 0, "j": 1, "f_min_ghz": f_lo, "f_max_ghz": f_hi,
             "target_db": -20.0, "weight": 5.0, "mode": "below"},
            {"i": 0, "j": 1, "f_min_ghz": 1.0, "f_max_ghz": max(1.0, f_lo - 2.0),
             "target_db": -3.0, "weight": 3.0, "mode": "above"},
            {"i": 0, "j": 1, "f_min_ghz": f_hi + 2.0, "f_max_ghz": 20.0,
             "target_db": -3.0, "weight": 3.0, "mode": "above"},
            {"i": 0, "j": 0, "f_min_ghz": 1.0, "f_max_ghz": max(1.0, f_lo - 2.0),
             "target_db": -8.0, "weight": 1.0, "mode": "below"},
        ]
        name = f"notch_{(f_lo+f_hi)/2:.0f}ghz"
        desc = f"Notch filter at {(f_lo+f_hi)/2:.1f} GHz"

    elif any(w in text_lower for w in ["lowpass", "low-pass", "low pass", "lpf"]):
        # Lowpass filter
        cutoff = freqs[0] if freqs else 10.0
        goals = [
            {"i": 0, "j": 1, "f_min_ghz": 1.0, "f_max_ghz": cutoff,
             "target_db": -3.0, "weight": 5.0, "mode": "above"},
            {"i": 0, "j": 0, "f_min_ghz": 1.0, "f_max_ghz": cutoff,
             "target_db": -10.0, "weight": 3.0, "mode": "below"},
            {"i": 0, "j": 1, "f_min_ghz": cutoff * 1.5, "f_max_ghz": 30.0,
             "target_db": -15.0, "weight": 2.0, "mode": "below"},
        ]
        name = f"lowpass_{cutoff:.0f}ghz"
        desc = f"Lowpass filter, {cutoff:.0f} GHz cutoff"

    elif any(w in text_lower for w in ["highpass", "high-pass", "high pass", "hpf"]):
        # Highpass filter
        cutoff = freqs[0] if freqs else 5.0
        goals = [
            {"i": 0, "j": 1, "f_min_ghz": cutoff, "f_max_ghz": 20.0,
             "target_db": -3.0, "weight": 5.0, "mode": "above"},
            {"i": 0, "j": 0, "f_min_ghz": cutoff, "f_max_ghz": 20.0,
             "target_db": -10.0, "weight": 3.0, "mode": "below"},
            {"i": 0, "j": 1, "f_min_ghz": 1.0, "f_max_ghz": max(1.0, cutoff * 0.5),
             "target_db": -15.0, "weight": 2.0, "mode": "below"},
        ]
        name = f"highpass_{cutoff:.0f}ghz"
        desc = f"Highpass filter, {cutoff:.0f} GHz cutoff"

    elif any(w in text_lower for w in ["hybrid", "3db coupler", "3 db coupler", "quadrature"]):
        # Hybrid coupler (3 dB equal split) — check before generic "coupler"
        f_center = freqs[0] if freqs else 10.0
        bw = 2.0
        goals = [
            {"i": 0, "j": 1, "f_min_ghz": f_center - bw, "f_max_ghz": f_center + bw,
             "target_db": -3.0, "weight": 5.0, "mode": "at"},
            {"i": 0, "j": 2, "f_min_ghz": f_center - bw, "f_max_ghz": f_center + bw,
             "target_db": -3.0, "weight": 5.0, "mode": "at"},
            {"i": 0, "j": 3, "f_min_ghz": f_center - bw, "f_max_ghz": f_center + bw,
             "target_db": -20.0, "weight": 3.0, "mode": "below"},
            {"i": 0, "j": 0, "f_min_ghz": f_center - bw, "f_max_ghz": f_center + bw,
             "target_db": -15.0, "weight": 2.0, "mode": "below"},
        ]
        name = f"hybrid_{f_center:.0f}ghz"
        desc = f"90° hybrid coupler at {f_center:.0f} GHz"
        return {"name": name, "description": desc, "goals": goals}

    elif any(w in text_lower for w in ["coupler", "directional", "coupling"]):
        # Directional coupler
        f_center = freqs[0] if freqs else 10.0
        coupling_db = dbs[0] if dbs else -10.0
        bw = 2.0
        goals = [
            {"i": 0, "j": 1, "f_min_ghz": f_center - bw, "f_max_ghz": f_center + bw,
             "target_db": -1.0, "weight": 3.0, "mode": "above"},
            {"i": 0, "j": 2, "f_min_ghz": f_center - bw, "f_max_ghz": f_center + bw,
             "target_db": coupling_db, "weight": 5.0, "mode": "at"},
            {"i": 0, "j": 3, "f_min_ghz": f_center - bw, "f_max_ghz": f_center + bw,
             "target_db": -30.0, "weight": 3.0, "mode": "below"},
            {"i": 0, "j": 0, "f_min_ghz": f_center - bw, "f_max_ghz": f_center + bw,
             "target_db": -15.0, "weight": 2.0, "mode": "below"},
        ]
        name = f"coupler_{f_center:.0f}ghz"
        desc = f"Directional coupler at {f_center:.0f} GHz, {coupling_db:.0f} dB coupling"
        return {"name": name, "description": desc, "goals": goals}

    elif any(w in text_lower for w in ["divider", "splitter", "power split", "wilkinson"]):
        # Power divider
        f_center = freqs[0] if freqs else 10.0
        bw = 2.0
        goals = [
            {"i": 0, "j": 1, "f_min_ghz": f_center - bw, "f_max_ghz": f_center + bw,
             "target_db": -3.5, "weight": 5.0, "mode": "above"},
            {"i": 0, "j": 2, "f_min_ghz": f_center - bw, "f_max_ghz": f_center + bw,
             "target_db": -3.5, "weight": 5.0, "mode": "above"},
            {"i": 1, "j": 2, "f_min_ghz": f_center - bw, "f_max_ghz": f_center + bw,
             "target_db": -15.0, "weight": 2.0, "mode": "below"},
            {"i": 0, "j": 0, "f_min_ghz": f_center - bw, "f_max_ghz": f_center + bw,
             "target_db": -15.0, "weight": 3.0, "mode": "below"},
        ]
        name = f"divider_{f_center:.0f}ghz"
        desc = f"Power divider at {f_center:.0f} GHz"
        return {"name": name, "description": desc, "goals": goals}

    elif any(w in text_lower for w in ["crossover", "cross-over"]):
        # Crossover (two through paths)
        f_center = freqs[0] if freqs else 10.0
        bw = 2.0
        goals = [
            {"i": 0, "j": 1, "f_min_ghz": f_center - bw, "f_max_ghz": f_center + bw,
             "target_db": -2.0, "weight": 5.0, "mode": "above"},
            {"i": 2, "j": 3, "f_min_ghz": f_center - bw, "f_max_ghz": f_center + bw,
             "target_db": -2.0, "weight": 5.0, "mode": "above"},
            {"i": 0, "j": 2, "f_min_ghz": f_center - bw, "f_max_ghz": f_center + bw,
             "target_db": -20.0, "weight": 3.0, "mode": "below"},
            {"i": 0, "j": 3, "f_min_ghz": f_center - bw, "f_max_ghz": f_center + bw,
             "target_db": -20.0, "weight": 3.0, "mode": "below"},
            {"i": 0, "j": 0, "f_min_ghz": f_center - bw, "f_max_ghz": f_center + bw,
             "target_db": -15.0, "weight": 2.0, "mode": "below"},
        ]
        name = f"crossover_{f_center:.0f}ghz"
        desc = f"Crossover at {f_center:.0f} GHz"
        return {"name": name, "description": desc, "goals": goals}

    elif any(w in text_lower for w in ["diplexer", "duplexer", "frequency split"]):
        # Diplexer
        if len(freqs) >= 2:
            f_lo, f_hi = sorted(freqs[:2])
        else:
            f_lo, f_hi = 5.0, 15.0
        goals = [
            {"i": 0, "j": 1, "f_min_ghz": f_lo - 2.0, "f_max_ghz": f_lo + 2.0,
             "target_db": -3.0, "weight": 5.0, "mode": "above"},
            {"i": 0, "j": 2, "f_min_ghz": f_hi - 2.0, "f_max_ghz": f_hi + 2.0,
             "target_db": -3.0, "weight": 5.0, "mode": "above"},
            {"i": 0, "j": 2, "f_min_ghz": f_lo - 2.0, "f_max_ghz": f_lo + 2.0,
             "target_db": -15.0, "weight": 2.0, "mode": "below"},
            {"i": 0, "j": 1, "f_min_ghz": f_hi - 2.0, "f_max_ghz": f_hi + 2.0,
             "target_db": -15.0, "weight": 2.0, "mode": "below"},
            {"i": 0, "j": 0, "f_min_ghz": f_lo - 2.0, "f_max_ghz": f_hi + 2.0,
             "target_db": -10.0, "weight": 2.0, "mode": "below"},
        ]
        name = f"diplexer_{f_lo:.0f}_{f_hi:.0f}ghz"
        desc = f"Diplexer {f_lo:.0f}/{f_hi:.0f} GHz"
        return {"name": name, "description": desc, "goals": goals}

    elif any(w in text_lower for w in ["antenna", "radiator", "patch", "slot"]):
        # Antenna (S11 match only)
        f_center = freqs[0] if freqs else 28.0
        bw = max(1.0, f_center * 0.1)
        goals = [
            {"i": 0, "j": 0, "f_min_ghz": f_center - bw, "f_max_ghz": f_center + bw,
             "target_db": -15.0, "weight": 5.0, "mode": "below"},
            {"i": 0, "j": 0, "f_min_ghz": f_center - bw * 1.5, "f_max_ghz": f_center + bw * 1.5,
             "target_db": -10.0, "weight": 3.0, "mode": "below"},
        ]
        name = f"antenna_{f_center:.0f}ghz"
        desc = f"Antenna at {f_center:.0f} GHz"
        return {"name": name, "description": desc, "goals": goals}

    else:
        # Default: broadband match
        f_lo = freqs[0] if len(freqs) >= 1 else 1.0
        f_hi = freqs[1] if len(freqs) >= 2 else 20.0
        if f_lo > f_hi:
            f_lo, f_hi = f_hi, f_lo
        goals = [
            {"i": 0, "j": 1, "f_min_ghz": f_lo, "f_max_ghz": f_hi,
             "target_db": -2.0, "weight": 5.0, "mode": "above"},
            {"i": 0, "j": 0, "f_min_ghz": f_lo, "f_max_ghz": f_hi,
             "target_db": -10.0, "weight": 3.0, "mode": "below"},
        ]
        name = f"broadband_{f_lo:.0f}_{f_hi:.0f}ghz"
        desc = f"Broadband match {f_lo:.0f}-{f_hi:.0f} GHz"

    # Always add isolation goals
    goals.extend([
        {"i": 0, "j": 2, "f_min_ghz": 1.0, "f_max_ghz": 30.0,
         "target_db": -20.0, "weight": 0.5, "mode": "below"},
        {"i": 0, "j": 3, "f_min_ghz": 1.0, "f_max_ghz": 30.0,
         "target_db": -20.0, "weight": 0.5, "mode": "below"},
    ])

    return {"name": name, "description": desc, "goals": goals}


def _parse_llm_response(response: str) -> dict:
    """Extract JSON from LLM response, handling markdown code blocks."""
    response = re.sub(r'```(?:json)?\s*', '', response)
    response = re.sub(r'```\s*$', '', response)
    match = re.search(r'\{.*\}', response, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON from LLM: {e}") from e
    raise ValueError(f"No JSON found in LLM response: {response[:200]}")


def _sanitize_goal(g: dict) -> dict | None:
    """Validate and sanitize a single goal dict. Returns None if unusable."""
    try:
        i = int(g.get("i", -1))
        j = int(g.get("j", -1))
        if not (0 <= i <= 3 and 0 <= j <= 3):
            return None

        f_min = float(g.get("f_min_ghz", g.get("f_min", 0)))
        f_max = float(g.get("f_max_ghz", g.get("f_max", 0)))
        f_min = max(0.1, min(50.0, f_min))
        f_max = max(0.1, min(50.0, f_max))
        if f_min >= f_max:
            return None

        target = float(g.get("target_db", g.get("target", 0)))
        target = max(-100.0, min(10.0, target))

        weight = max(0.1, min(20.0, float(g.get("weight", 1.0))))

        mode = str(g.get("mode", "below")).strip().lower()
        if mode not in ("below", "above", "at"):
            if mode in ("min", "less", "lt", "<", "suppress", "reject"):
                mode = "below"
            elif mode in ("max", "greater", "gt", ">", "pass", "through"):
                mode = "above"
            elif mode in ("equal", "eq", "=", "match", "exact"):
                mode = "at"
            else:
                mode = "below"

        return {
            "i": i, "j": j,
            "f_min_ghz": round(f_min, 2), "f_max_ghz": round(f_max, 2),
            "target_db": round(target, 1), "weight": round(weight, 1),
            "mode": mode,
        }
    except (ValueError, TypeError, AttributeError):
        return None


def extract_objective(
    text: str,
    backend: str = "fallback",
    model: str = "",
    base_url: str = "http://localhost:11434",
    api_key: str = "",
) -> tuple[MatchingObjective, dict]:
    """Extract a MatchingObjective from natural language.

    Returns (objective, raw_params_dict) for display.
    Raises PromptRejectedError if the LLM determines this isn't an RF task.
    """
    if backend == "fallback":
        params = _fallback_extract(text)
    elif backend == "llama":
        checkpoint_dir = model if model else DEFAULT_LLAMA_DIR
        raw = _call_llama(text, checkpoint_dir=checkpoint_dir)
        params = _parse_llm_response(raw)
    elif backend == "ollama":
        raw = _call_ollama(text, model=model or "llama3.2", base_url=base_url)
        params = _parse_llm_response(raw)
    else:
        raise ValueError(f"Unknown backend: {backend}")

    if params.get("rejected"):
        raise PromptRejectedError(
            params.get("reason", "This doesn't appear to be an RF/microwave design request.")
        )

    raw_goals = params.get("goals")
    if not raw_goals or not isinstance(raw_goals, list):
        raise PromptRejectedError("Could not extract any S-parameter goals from the description.")

    clean_goals = [_sanitize_goal(g) for g in raw_goals if isinstance(g, dict)]
    clean_goals = [g for g in clean_goals if g is not None]
    if not clean_goals:
        raise PromptRejectedError(
            "LLM returned goals but none were valid (bad ports, freq, or types). "
            "Try rephrasing or use a preset."
        )

    params["goals"] = clean_goals

    goals = [
        SParamGoal(
            i=g["i"], j=g["j"],
            f_min_ghz=g["f_min_ghz"], f_max_ghz=g["f_max_ghz"],
            target_db=g["target_db"],
            weight=g["weight"], mode=g["mode"],
        )
        for g in clean_goals
    ]

    objective = MatchingObjective(
        name=params.get("name", "custom"),
        description=params.get("description", text[:80]),
        goals=goals,
    )

    return objective, params
