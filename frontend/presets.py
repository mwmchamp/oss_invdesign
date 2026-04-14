"""Canonical preset specs exposed as one-click starting points in the UI.

Two presets, matching the report description:
  1. Broadband matching network  — flat through-path 10-15 GHz, low return loss
  2. 15 GHz high-pass filter     — pass > 15 GHz, reject below
"""

from __future__ import annotations


PRESETS: dict[str, dict] = {
    "Broadband matching network": {
        "prompt": "Broadband matching network: flat through-path between N "
                  "and S from 10 to 15 GHz, return loss better than -10 dB, "
                  "with low leakage to E and W.",
        "goals": [
            {"i": 0, "j": 1, "f_min_ghz": 10.0, "f_max_ghz": 15.0,
             "target_db": -3.0, "weight": 5.0, "mode": "above"},
            {"i": 0, "j": 0, "f_min_ghz": 10.0, "f_max_ghz": 15.0,
             "target_db": -10.0, "weight": 3.0, "mode": "below"},
            {"i": 0, "j": 2, "f_min_ghz": 1.0, "f_max_ghz": 30.0,
             "target_db": -20.0, "weight": 0.5, "mode": "below"},
            {"i": 0, "j": 3, "f_min_ghz": 1.0, "f_max_ghz": 30.0,
             "target_db": -20.0, "weight": 0.5, "mode": "below"},
        ],
    },
    "15 GHz high-pass filter": {
        "prompt": "15 GHz high-pass filter: pass energy above 15 GHz from N "
                  "to S (insertion loss better than -3 dB), reject below "
                  "10 GHz by at least 15 dB.",
        "goals": [
            {"i": 0, "j": 1, "f_min_ghz": 15.0, "f_max_ghz": 25.0,
             "target_db": -3.0, "weight": 5.0, "mode": "above"},
            {"i": 0, "j": 0, "f_min_ghz": 15.0, "f_max_ghz": 25.0,
             "target_db": -10.0, "weight": 3.0, "mode": "below"},
            {"i": 0, "j": 1, "f_min_ghz": 1.0, "f_max_ghz": 10.0,
             "target_db": -15.0, "weight": 3.0, "mode": "below"},
            {"i": 0, "j": 2, "f_min_ghz": 1.0, "f_max_ghz": 30.0,
             "target_db": -20.0, "weight": 0.5, "mode": "below"},
            {"i": 0, "j": 3, "f_min_ghz": 1.0, "f_max_ghz": 30.0,
             "target_db": -20.0, "weight": 0.5, "mode": "below"},
        ],
    },
}

PRESET_NAMES = list(PRESETS.keys())


def apply_preset(preset_name: str):
    """Return (prompt, goals_df, match_df) from a preset key.

    Used by both Natural-Language preset buttons (which want the prompt back
    in the textbox) and JSON-upload preset buttons (which just want to fill
    the goals/match tables).
    """
    from frontend.goals_io import goals_to_df, matches_to_df, empty_match_df

    preset = PRESETS.get(preset_name)
    if preset is None:
        from frontend.goals_io import empty_goals_df
        return "", empty_goals_df(), empty_match_df()
    goals_df = goals_to_df(preset["goals"])
    match_df = (matches_to_df(preset["match"])
                if preset.get("match") else empty_match_df())
    return preset["prompt"], goals_df, match_df
