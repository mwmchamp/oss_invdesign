"""Goal table ↔ DataFrame ↔ JSON schema conversion.

Schema used by the JSON-upload bypass:
    {
        "name": "my device",
        "description": "optional free text",
        "sparam_goals": [
            {"i": "N", "j": "S", "f_min_ghz": 10, "f_max_ghz": 15,
             "target_db": -3.0, "weight": 5.0, "mode": "above"}
        ],
        "impedance_goals": [            # optional
            {"z_source": [50, 0], "z_load": [25, 25],
             "f_min_ghz": 11, "f_max_ghz": 13,
             "in_port": "N", "out_port": "S", "weight": 10.0}
        ]
    }
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from optimizer.objectives import (
    MatchingObjective, SParamGoal,
    ImpedanceMatchObjective, ImpedanceMatchGoal,
)

PORTS = ["N", "S", "E", "W"]
PORT_IDX = {p: i for i, p in enumerate(PORTS)}
PORT_LABEL = {i: p for p, i in PORT_IDX.items()}

GOAL_COLUMNS = ["i", "j", "f_min_ghz", "f_max_ghz",
                "target_db", "weight", "mode"]
MATCH_COLUMNS = ["Re(Zs)", "Im(Zs)", "Re(Zl)", "Im(Zl)",
                 "f_min_ghz", "f_max_ghz",
                 "in_port", "out_port", "weight"]

# Mode legend — rendered as a markdown caption above the goal table.
MODE_LEGEND_MD = (
    "**Mode:** 🟢 `above` (minimum bound, ≥ target dB) · "
    "🔴 `below` (maximum bound, ≤ target dB) · "
    "🟡 `at` (exact target dB)"
)
VALID_MODES = ("above", "below", "at")


def _is_num(v) -> bool:
    try:
        float(v)
        return True
    except (ValueError, TypeError):
        return False


def _port_from(v, default: str = "N") -> int:
    if _is_num(v):
        return int(v)
    return PORT_IDX[str(v).strip().upper()]


def empty_goals_df() -> pd.DataFrame:
    return pd.DataFrame(columns=GOAL_COLUMNS)


def empty_match_df() -> pd.DataFrame:
    return pd.DataFrame(columns=MATCH_COLUMNS)


def goals_to_df(goals: list[dict]) -> pd.DataFrame:
    rows = []
    for g in goals:
        rows.append({
            "i": PORT_LABEL[g["i"]] if _is_num(g["i"]) else str(g["i"]).upper(),
            "j": PORT_LABEL[g["j"]] if _is_num(g["j"]) else str(g["j"]).upper(),
            "f_min_ghz": float(g["f_min_ghz"]),
            "f_max_ghz": float(g["f_max_ghz"]),
            "target_db": float(g["target_db"]),
            "weight": float(g["weight"]),
            "mode": str(g.get("mode", "below")),
        })
    return pd.DataFrame(rows, columns=GOAL_COLUMNS) if rows else empty_goals_df()


def matches_to_df(matches: list[dict]) -> pd.DataFrame:
    rows = []
    for m in matches:
        zs = m.get("z_source", [50.0, 0.0])
        zl = m.get("z_load", [50.0, 0.0])
        rows.append({
            "Re(Zs)": float(zs[0]), "Im(Zs)": float(zs[1]),
            "Re(Zl)": float(zl[0]), "Im(Zl)": float(zl[1]),
            "f_min_ghz": float(m["f_min_ghz"]),
            "f_max_ghz": float(m["f_max_ghz"]),
            "in_port": str(m.get("in_port", "N")).upper()
                if not _is_num(m.get("in_port", "N"))
                else PORT_LABEL[int(m["in_port"])],
            "out_port": str(m.get("out_port", "S")).upper()
                if not _is_num(m.get("out_port", "S"))
                else PORT_LABEL[int(m["out_port"])],
            "weight": float(m.get("weight", 10.0)),
        })
    return pd.DataFrame(rows, columns=MATCH_COLUMNS) if rows else empty_match_df()


def df_to_sparam_goals(df) -> list[SParamGoal]:
    if df is None or len(df) == 0:
        return []
    rows = df.to_dict("records") if hasattr(df, "to_dict") else list(df)
    out = []
    for r in rows:
        try:
            i = _port_from(r.get("i", "N"))
            j = _port_from(r.get("j", "S"))
            mode = str(r.get("mode", "below")).strip().lower()
            if mode not in VALID_MODES:
                continue
            fmin = float(r["f_min_ghz"]); fmax = float(r["f_max_ghz"])
            if not (0 < fmin < fmax):
                continue
            out.append(SParamGoal(
                i=i, j=j,
                f_min_ghz=fmin, f_max_ghz=fmax,
                target_db=float(r["target_db"]),
                weight=float(r.get("weight", 1.0)),
                mode=mode,
            ))
        except (KeyError, ValueError, TypeError):
            continue
    return out


def df_to_impedance_goals(df) -> list[ImpedanceMatchGoal]:
    if df is None or len(df) == 0:
        return []
    rows = df.to_dict("records") if hasattr(df, "to_dict") else list(df)
    out = []
    for r in rows:
        try:
            zs = complex(float(r["Re(Zs)"]), float(r["Im(Zs)"]))
            zl = complex(float(r["Re(Zl)"]), float(r["Im(Zl)"]))
            fmin = float(r["f_min_ghz"]); fmax = float(r["f_max_ghz"])
            if not (0 < fmin < fmax):
                continue
            ip = _port_from(r.get("in_port", "N"))
            op = _port_from(r.get("out_port", "S"))
            out.append(ImpedanceMatchGoal(
                z_source=zs, z_load=zl,
                f_min_ghz=fmin, f_max_ghz=fmax,
                weight=float(r.get("weight", 10.0)),
                in_port=ip, out_port=op,
            ))
        except (KeyError, ValueError, TypeError):
            continue
    return out


def build_objective(goals_df, match_df, name: str = "custom"):
    sgs = df_to_sparam_goals(goals_df)
    igs = df_to_impedance_goals(match_df)
    if igs:
        return ImpedanceMatchObjective(
            name=name, description="User-defined impedance match",
            goals=igs, sparam_goals=sgs,
        )
    if sgs:
        return MatchingObjective(
            name=name, description="User-defined S-parameter spec",
            goals=sgs,
        )
    return None


# ── JSON upload ────────────────────────────────────────────────────────────

def parse_specs_json(file_obj) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """Read a specs JSON file and return (goals_df, match_df, status).

    `file_obj` is the Gradio File component output: either a file path (str)
    or an object with .name. The schema is documented at module top.
    """
    if file_obj is None:
        return empty_goals_df(), empty_match_df(), "No file selected."
    path = Path(file_obj if isinstance(file_obj, str) else file_obj.name)
    try:
        data = json.loads(path.read_text())
    except Exception as e:
        return empty_goals_df(), empty_match_df(), f"❌ Could not parse JSON: {e}"

    sgs = data.get("sparam_goals") or data.get("goals") or []
    mgs = data.get("impedance_goals") or data.get("matches") or []
    if not isinstance(sgs, list) or not isinstance(mgs, list):
        return empty_goals_df(), empty_match_df(), (
            "❌ Invalid schema: expected `sparam_goals` and/or "
            "`impedance_goals` as lists."
        )
    name = data.get("name") or path.stem
    return (
        goals_to_df(sgs),
        matches_to_df(mgs),
        f"✓ Loaded '{name}' · {len(sgs)} S-param goal(s), "
        f"{len(mgs)} impedance goal(s)",
    )
