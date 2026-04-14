"""Matching network objective specifications.

Defines target S-parameter goals for common RF matching problems.
Each objective specifies frequency bands, target |S| values, and weights.

Port convention: N=0, S=1, E=2, W=3
  - For 2-port matching: typically use ports 0 (N) and 1 (S) as signal ports
  - Ports 2 (E) and 3 (W) should be isolated (S_x2, S_x3 small)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class SParamGoal:
    """A single S-parameter target over a frequency band.

    Attributes:
        i, j: port indices (0-3)
        f_min_ghz, f_max_ghz: frequency band
        target_db: target |S_ij| in dB
        weight: importance weight in fitness function
        mode: 'below' (S_ij < target), 'above' (S_ij > target), or 'at' (S_ij ≈ target)
    """
    i: int
    j: int
    f_min_ghz: float
    f_max_ghz: float
    target_db: float
    weight: float = 1.0
    mode: str = "below"  # "below", "above", or "at"


@dataclass
class MatchingObjective:
    """Complete matching network specification.

    Defines what the optimizer should achieve in terms of S-parameters.
    """
    name: str
    description: str
    goals: list[SParamGoal]
    freq_ghz: np.ndarray = field(default_factory=lambda: np.linspace(1, 30, 30))

    def evaluate(self, sparams_cpx: np.ndarray, freq_ghz: np.ndarray | None = None) -> dict:
        """Score S-parameters against all goals.

        Args:
            sparams_cpx: (n_freq, 4, 4) complex S-parameters
            freq_ghz: frequency array matching sparams_cpx. If None, uses
                self.freq_ghz (must have same length as sparams_cpx axis 0).

        Returns:
            dict with 'fitness' (higher=better), 'goal_scores', 'details'
        """
        freqs = freq_ghz if freq_ghz is not None else self.freq_ghz
        n_freq = sparams_cpx.shape[0]
        if len(freqs) != n_freq:
            raise ValueError(
                f"freq_ghz length ({len(freqs)}) != sparams freq axis ({n_freq}). "
                f"Pass freq_ghz explicitly if sparams has non-standard frequency grid."
            )
        total_penalty = 0.0
        goal_details = []

        for goal in self.goals:
            # Find frequency indices in band
            f_mask = (freqs >= goal.f_min_ghz) & (freqs <= goal.f_max_ghz)
            if not f_mask.any():
                goal_details.append({"goal": f"S{goal.i+1}{goal.j+1}", "penalty": 0.0,
                                     "achieved_db": float("nan")})
                continue

            # Extract |S_ij| in dB over the band
            sij = sparams_cpx[f_mask, goal.i, goal.j]
            mag_db = 20 * np.log10(np.abs(sij) + 1e-12)

            if goal.mode == "below":
                # Penalize where mag > target (want S_ij below target)
                violations = np.maximum(mag_db - goal.target_db, 0)
                penalty = goal.weight * np.mean(violations ** 2)
            elif goal.mode == "above":
                # Penalize where mag < target (want S_ij above target)
                violations = np.maximum(goal.target_db - mag_db, 0)
                penalty = goal.weight * np.mean(violations ** 2)
            else:  # "at"
                penalty = goal.weight * np.mean((mag_db - goal.target_db) ** 2)

            total_penalty += penalty
            goal_details.append({
                "goal": f"|S{goal.i+1}{goal.j+1}| {goal.mode} {goal.target_db:.0f}dB "
                        f"@ {goal.f_min_ghz:.0f}-{goal.f_max_ghz:.0f}GHz",
                "penalty": float(penalty),
                "achieved_db": float(np.mean(mag_db)),
                "worst_db": float(np.max(mag_db) if goal.mode == "below"
                                  else np.min(mag_db)),
            })

        # Fitness = negative penalty (higher is better, 0 = perfect)
        fitness = -total_penalty
        return {
            "fitness": float(fitness),
            "penalty": float(total_penalty),
            "goals": goal_details,
        }


# ── Impedance matching efficiency objective ───────────────────────────────


@dataclass
class ImpedanceMatchGoal:
    """A matching efficiency target at specific frequencies.

    Computes transmitted power T through a 2-port S-parameter network
    with given source and load impedances:

        T = |S21|²(1-|Γl|²)(1-|Γs|²) / |(1-Γs·S11)(1-Γl·S22) - S21·Γl·S12·Γs|²

    where Γs = (Zs-50)/(Zs+50), Γl = (Zl-50)/(Zl+50).

    Attributes:
        z_source: source impedance (complex, per freq or scalar)
        z_load: load impedance (complex, per freq or scalar)
        f_min_ghz, f_max_ghz: frequency band
        weight: importance weight
        in_port, out_port: port indices for signal path (default 0→1)
    """
    z_source: complex
    z_load: complex
    f_min_ghz: float
    f_max_ghz: float
    weight: float = 1.0
    in_port: int = 0
    out_port: int = 1


@dataclass
class ImpedanceMatchObjective:
    """Matching network objective using transmitted power (matching efficiency).

    This is the standard RF matching network cost function: maximize power
    transfer from source Zs to load Zl through the 2-port network.

    The cost drives sqrt(T) toward 1 (perfect match) at each frequency:
        cost = Σ_f w_f · (1 - sqrt(T_f))²

    Reference: ga_matching_network.ipynb (Karahan/Sengupta group).
    """
    name: str
    description: str
    goals: list[ImpedanceMatchGoal]
    freq_ghz: np.ndarray = field(default_factory=lambda: np.linspace(1, 30, 30))
    # Also include S-param goals for isolation etc.
    sparam_goals: list[SParamGoal] = field(default_factory=list)

    def evaluate(self, sparams_cpx: np.ndarray, freq_ghz: np.ndarray | None = None) -> dict:
        """Score S-parameters for impedance matching efficiency.

        Args:
            sparams_cpx: (n_freq, 4, 4) complex S-parameters
            freq_ghz: frequency array. If None, uses self.freq_ghz.

        Returns:
            dict with 'fitness', 'penalty', 'goals', 'efficiency_db'
        """
        freqs = freq_ghz if freq_ghz is not None else self.freq_ghz
        n_freq = sparams_cpx.shape[0]
        if len(freqs) != n_freq:
            raise ValueError(
                f"freq_ghz length ({len(freqs)}) != sparams freq axis ({n_freq})"
            )

        total_penalty = 0.0
        goal_details = []

        for goal in self.goals:
            f_mask = (freqs >= goal.f_min_ghz) & (freqs <= goal.f_max_ghz)
            if not f_mask.any():
                goal_details.append({
                    "goal": f"match Z{goal.in_port+1}→Z{goal.out_port+1}",
                    "penalty": 0.0, "achieved_db": float("nan"),
                })
                continue

            # Compute matching efficiency at each frequency in band
            s_band = sparams_cpx[f_mask]
            gamma_s = (goal.z_source - 50) / (goal.z_source + 50)
            gamma_l = (goal.z_load - 50) / (goal.z_load + 50)

            ip, op = goal.in_port, goal.out_port
            S11 = s_band[:, ip, ip]
            S22 = s_band[:, op, op]
            S21 = s_band[:, op, ip]
            S12 = s_band[:, ip, op]

            denom = np.abs(
                (1 - gamma_s * S11) * (1 - gamma_l * S22)
                - S21 * gamma_l * S12 * gamma_s
            ) ** 2
            denom = np.maximum(denom, 1e-30)

            T = (np.abs(S21) ** 2
                 * (1 - np.abs(gamma_l) ** 2)
                 * (1 - np.abs(gamma_s) ** 2)
                 / denom)
            T = np.clip(T, 0, 1)  # clamp for numerical safety

            # Cost: (1 - sqrt(T))^2, want T → 1
            eff = np.sqrt(T)
            cost = np.mean((1 - eff) ** 2)
            penalty = goal.weight * cost

            eff_db = 10 * np.log10(T + 1e-30)
            total_penalty += penalty
            goal_details.append({
                "goal": f"match Zs={goal.z_source}→Zl={goal.z_load} "
                        f"@ {goal.f_min_ghz:.0f}-{goal.f_max_ghz:.0f}GHz",
                "penalty": float(penalty),
                "achieved_db": float(np.mean(eff_db)),
                "worst_db": float(np.min(eff_db)),
                "mean_efficiency": float(np.mean(eff)),
            })

        # Also evaluate any S-param goals (isolation, return loss, etc.)
        for goal in self.sparam_goals:
            f_mask = (freqs >= goal.f_min_ghz) & (freqs <= goal.f_max_ghz)
            if not f_mask.any():
                continue
            sij = sparams_cpx[f_mask, goal.i, goal.j]
            mag_db = 20 * np.log10(np.abs(sij) + 1e-12)
            if goal.mode == "below":
                violations = np.maximum(mag_db - goal.target_db, 0)
                penalty = goal.weight * np.mean(violations ** 2)
            elif goal.mode == "above":
                violations = np.maximum(goal.target_db - mag_db, 0)
                penalty = goal.weight * np.mean(violations ** 2)
            else:
                penalty = goal.weight * np.mean((mag_db - goal.target_db) ** 2)
            total_penalty += penalty
            goal_details.append({
                "goal": f"|S{goal.i+1}{goal.j+1}| {goal.mode} {goal.target_db:.0f}dB "
                        f"@ {goal.f_min_ghz:.0f}-{goal.f_max_ghz:.0f}GHz",
                "penalty": float(penalty),
                "achieved_db": float(np.mean(mag_db)),
                "worst_db": float(np.max(mag_db) if goal.mode == "below"
                                  else np.min(mag_db)),
            })

        fitness = -total_penalty
        return {
            "fitness": float(fitness),
            "penalty": float(total_penalty),
            "goals": goal_details,
        }


# ── Pre-defined matching network objectives ──────────────────────────────

def bandpass_5ghz() -> MatchingObjective:
    """5 GHz bandpass matching network (N↔S ports).

    Goal: pass 4.5-5.5 GHz between ports 0 and 1, reject elsewhere.
    Typical use: Wi-Fi 5 GHz band matching.
    """
    return MatchingObjective(
        name="bandpass_5ghz",
        description="5 GHz bandpass (N↔S), 4.5-5.5 GHz passband",
        goals=[
            # Passband: low insertion loss
            SParamGoal(0, 1, 4.5, 5.5, target_db=-3.0, weight=5.0, mode="above"),
            # Passband: good return loss
            SParamGoal(0, 0, 4.5, 5.5, target_db=-10.0, weight=3.0, mode="below"),
            # Lower stopband rejection
            SParamGoal(0, 1, 1.0, 3.5, target_db=-15.0, weight=1.0, mode="below"),
            # Upper stopband rejection
            SParamGoal(0, 1, 7.0, 15.0, target_db=-15.0, weight=1.0, mode="below"),
            # Isolation to other ports
            SParamGoal(0, 2, 1.0, 30.0, target_db=-20.0, weight=0.5, mode="below"),
            SParamGoal(0, 3, 1.0, 30.0, target_db=-20.0, weight=0.5, mode="below"),
        ],
    )


def lowpass_10ghz() -> MatchingObjective:
    """10 GHz lowpass matching network (N↔S ports).

    Goal: pass DC-10 GHz, reject above 15 GHz.
    """
    return MatchingObjective(
        name="lowpass_10ghz",
        description="10 GHz lowpass (N↔S), DC-10 GHz passband",
        goals=[
            # Passband: low insertion loss
            SParamGoal(0, 1, 1.0, 10.0, target_db=-3.0, weight=5.0, mode="above"),
            # Passband: good return loss
            SParamGoal(0, 0, 1.0, 10.0, target_db=-10.0, weight=3.0, mode="below"),
            # Stopband rejection
            SParamGoal(0, 1, 15.0, 30.0, target_db=-15.0, weight=2.0, mode="below"),
            # Isolation
            SParamGoal(0, 2, 1.0, 30.0, target_db=-20.0, weight=0.5, mode="below"),
            SParamGoal(0, 3, 1.0, 30.0, target_db=-20.0, weight=0.5, mode="below"),
        ],
    )


def broadband_match() -> MatchingObjective:
    """Broadband impedance match (N↔S ports).

    Goal: low return loss across 1-20 GHz, maximize power transfer.
    Useful as a broadband matching network or interconnect.
    """
    return MatchingObjective(
        name="broadband_match",
        description="Broadband match (N↔S), 1-20 GHz",
        goals=[
            # Low insertion loss across band
            SParamGoal(0, 1, 1.0, 20.0, target_db=-2.0, weight=5.0, mode="above"),
            # Good return loss
            SParamGoal(0, 0, 1.0, 20.0, target_db=-10.0, weight=3.0, mode="below"),
            # Isolation
            SParamGoal(0, 2, 1.0, 30.0, target_db=-20.0, weight=0.5, mode="below"),
            SParamGoal(0, 3, 1.0, 30.0, target_db=-20.0, weight=0.5, mode="below"),
        ],
    )


def notch_10ghz() -> MatchingObjective:
    """10 GHz notch filter (N↔S ports).

    Goal: reject 9-11 GHz, pass everything else.
    """
    return MatchingObjective(
        name="notch_10ghz",
        description="10 GHz notch (N↔S), reject 9-11 GHz",
        goals=[
            # Notch: high rejection in band
            SParamGoal(0, 1, 9.0, 11.0, target_db=-20.0, weight=5.0, mode="below"),
            # Passband below notch
            SParamGoal(0, 1, 1.0, 7.0, target_db=-3.0, weight=3.0, mode="above"),
            # Passband above notch
            SParamGoal(0, 1, 13.0, 20.0, target_db=-3.0, weight=3.0, mode="above"),
            # Return loss outside notch
            SParamGoal(0, 0, 1.0, 7.0, target_db=-8.0, weight=1.0, mode="below"),
        ],
    )


# ── 4-port structures: couplers, dividers, crossovers ──────────────────


def directional_coupler_10ghz() -> MatchingObjective:
    """10 GHz directional coupler.

    Port mapping: 0(N)=input, 1(S)=through, 2(E)=coupled, 3(W)=isolated.
    Target: -10 dB coupling at 10 GHz with >20 dB directivity.
    """
    return MatchingObjective(
        name="coupler_10ghz",
        description="10 GHz directional coupler, -10 dB coupling",
        goals=[
            # Through port: low insertion loss
            SParamGoal(0, 1, 8.0, 12.0, target_db=-1.0, weight=3.0, mode="above"),
            # Coupled port: target coupling level
            SParamGoal(0, 2, 8.0, 12.0, target_db=-10.0, weight=5.0, mode="at"),
            # Isolated port: high isolation (directivity)
            SParamGoal(0, 3, 8.0, 12.0, target_db=-30.0, weight=3.0, mode="below"),
            # Input return loss
            SParamGoal(0, 0, 8.0, 12.0, target_db=-15.0, weight=2.0, mode="below"),
        ],
    )


def hybrid_coupler_10ghz() -> MatchingObjective:
    """10 GHz 90-degree hybrid (3 dB) coupler.

    Port mapping: 0(N)=input, 1(S)=through (-3dB, 0°), 2(E)=coupled (-3dB, -90°), 3(W)=isolated.
    Equal power split with 90° phase difference.
    """
    return MatchingObjective(
        name="hybrid_10ghz",
        description="10 GHz 90° hybrid coupler, equal split",
        goals=[
            # Through: -3 dB (equal split)
            SParamGoal(0, 1, 8.0, 12.0, target_db=-3.0, weight=5.0, mode="at"),
            # Coupled: -3 dB (equal split)
            SParamGoal(0, 2, 8.0, 12.0, target_db=-3.0, weight=5.0, mode="at"),
            # Isolated: high isolation
            SParamGoal(0, 3, 8.0, 12.0, target_db=-20.0, weight=3.0, mode="below"),
            # Input match
            SParamGoal(0, 0, 8.0, 12.0, target_db=-15.0, weight=2.0, mode="below"),
        ],
    )


def power_divider_10ghz() -> MatchingObjective:
    """10 GHz 2-way power divider.

    Port mapping: 0(N)=input, 1(S)=output1, 2(E)=output2, 3(W)=isolated/terminated.
    Equal split between ports 1 and 2.
    """
    return MatchingObjective(
        name="divider_10ghz",
        description="10 GHz 2-way power divider",
        goals=[
            # Output 1: -3 dB (equal split)
            SParamGoal(0, 1, 8.0, 12.0, target_db=-3.5, weight=5.0, mode="above"),
            # Output 2: -3 dB (equal split)
            SParamGoal(0, 2, 8.0, 12.0, target_db=-3.5, weight=5.0, mode="above"),
            # Output isolation
            SParamGoal(1, 2, 8.0, 12.0, target_db=-15.0, weight=2.0, mode="below"),
            # Input match
            SParamGoal(0, 0, 8.0, 12.0, target_db=-15.0, weight=3.0, mode="below"),
            # Port 3 isolated
            SParamGoal(0, 3, 1.0, 30.0, target_db=-20.0, weight=1.0, mode="below"),
        ],
    )


def crossover_10ghz() -> MatchingObjective:
    """10 GHz crossover: two crossing signal paths.

    Port mapping: 0(N)↔1(S) and 2(E)↔3(W) as two independent through paths.
    Both paths low loss, high isolation between paths.
    """
    return MatchingObjective(
        name="crossover_10ghz",
        description="10 GHz crossover, two independent paths",
        goals=[
            # Path 1: N→S through
            SParamGoal(0, 1, 8.0, 12.0, target_db=-2.0, weight=5.0, mode="above"),
            # Path 2: E→W through
            SParamGoal(2, 3, 8.0, 12.0, target_db=-2.0, weight=5.0, mode="above"),
            # Cross-path isolation
            SParamGoal(0, 2, 8.0, 12.0, target_db=-20.0, weight=3.0, mode="below"),
            SParamGoal(0, 3, 8.0, 12.0, target_db=-20.0, weight=3.0, mode="below"),
            # Return loss both paths
            SParamGoal(0, 0, 8.0, 12.0, target_db=-15.0, weight=2.0, mode="below"),
            SParamGoal(2, 2, 8.0, 12.0, target_db=-15.0, weight=2.0, mode="below"),
        ],
    )


def diplexer_5_15ghz() -> MatchingObjective:
    """5/15 GHz diplexer: route low-band to port 1, high-band to port 2.

    Port mapping: 0(N)=input, 1(S)=low-band output, 2(E)=high-band output.
    """
    return MatchingObjective(
        name="diplexer_5_15ghz",
        description="5/15 GHz diplexer, split low/high band",
        goals=[
            # Low band → port 1
            SParamGoal(0, 1, 3.0, 7.0, target_db=-3.0, weight=5.0, mode="above"),
            # High band → port 2
            SParamGoal(0, 2, 13.0, 17.0, target_db=-3.0, weight=5.0, mode="above"),
            # Low band rejected from port 2
            SParamGoal(0, 2, 3.0, 7.0, target_db=-15.0, weight=2.0, mode="below"),
            # High band rejected from port 1
            SParamGoal(0, 1, 13.0, 17.0, target_db=-15.0, weight=2.0, mode="below"),
            # Input match
            SParamGoal(0, 0, 3.0, 17.0, target_db=-10.0, weight=2.0, mode="below"),
            # Port 3 isolated
            SParamGoal(0, 3, 1.0, 30.0, target_db=-20.0, weight=0.5, mode="below"),
        ],
    )


def antenna_match_28ghz() -> MatchingObjective:
    """28 GHz antenna impedance match (single-port S11 optimization).

    For use with antenna simulation mode (no ground plane).
    Only optimizes S11 — radiation pattern depends on structure.
    """
    return MatchingObjective(
        name="antenna_28ghz",
        description="28 GHz antenna, S11 match",
        goals=[
            # Deep S11 null at 28 GHz
            SParamGoal(0, 0, 26.0, 30.0, target_db=-15.0, weight=5.0, mode="below"),
            # Reasonable bandwidth
            SParamGoal(0, 0, 25.0, 30.0, target_db=-10.0, weight=3.0, mode="below"),
            # Other ports isolated (if present)
            SParamGoal(0, 1, 1.0, 30.0, target_db=-20.0, weight=0.5, mode="below"),
            SParamGoal(0, 2, 1.0, 30.0, target_db=-20.0, weight=0.5, mode="below"),
        ],
    )


# ── Pre-defined impedance matching objectives ────────────────────────────


def narrowband_match_10ghz() -> ImpedanceMatchObjective:
    """Narrowband impedance match at 10 GHz.

    Match 50Ω source to (25+50j)Ω load at 10 GHz.
    Typical use: single-frequency LNA input matching.
    """
    return ImpedanceMatchObjective(
        name="narrowband_match_10ghz",
        description="10 GHz narrowband match, Zs=50Ω → Zl=(25+50j)Ω",
        goals=[
            ImpedanceMatchGoal(
                z_source=50+0j, z_load=25+50j,
                f_min_ghz=9.0, f_max_ghz=11.0, weight=10.0,
            ),
        ],
        sparam_goals=[
            SParamGoal(0, 2, 1.0, 30.0, target_db=-20.0, weight=0.5, mode="below"),
            SParamGoal(0, 3, 1.0, 30.0, target_db=-20.0, weight=0.5, mode="below"),
        ],
    )


def wideband_match_5_15ghz() -> ImpedanceMatchObjective:
    """Wideband impedance match from 5-15 GHz.

    Match 50Ω source to (30+20j)Ω load across a wide band.
    """
    return ImpedanceMatchObjective(
        name="wideband_match_5_15ghz",
        description="5-15 GHz wideband match, Zs=50Ω → Zl=(30+20j)Ω",
        goals=[
            ImpedanceMatchGoal(
                z_source=50+0j, z_load=30+20j,
                f_min_ghz=5.0, f_max_ghz=15.0, weight=10.0,
            ),
        ],
        sparam_goals=[
            SParamGoal(0, 0, 5.0, 15.0, target_db=-10.0, weight=2.0, mode="below"),
            SParamGoal(0, 2, 1.0, 30.0, target_db=-20.0, weight=0.5, mode="below"),
            SParamGoal(0, 3, 1.0, 30.0, target_db=-20.0, weight=0.5, mode="below"),
        ],
    )


# Registry
OBJECTIVES = {
    # S-param magnitude objectives (filters, couplers)
    "bandpass_5ghz": bandpass_5ghz,
    "lowpass_10ghz": lowpass_10ghz,
    "broadband_match": broadband_match,
    "notch_10ghz": notch_10ghz,
    # 4-port structures
    "coupler_10ghz": directional_coupler_10ghz,
    "hybrid_10ghz": hybrid_coupler_10ghz,
    "divider_10ghz": power_divider_10ghz,
    "crossover_10ghz": crossover_10ghz,
    "diplexer_5_15ghz": diplexer_5_15ghz,
    # Antenna
    "antenna_28ghz": antenna_match_28ghz,
    # Impedance matching (transmitted power)
    "narrowband_match_10ghz": narrowband_match_10ghz,
    "wideband_match_5_15ghz": wideband_match_5_15ghz,
}
