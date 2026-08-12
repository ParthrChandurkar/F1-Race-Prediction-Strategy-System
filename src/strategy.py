"""
src/strategy.py
Professional F1 strategy recommendation engine.
Inputs: circuit, grid position, tyre choice, weather.
Outputs: full multi-stop strategy with lap windows, risk analysis, alternatives.
"""

import os
import math
import pandas as pd
from src.f1_2024_data import CIRCUIT_CHARACTERISTICS, CIRCUIT_DATA_MAP

COMPOUND_DEG = {
    "Soft":         {"laps_peak": 12, "laps_max": 20, "pace_delta": 0.0,  "color": "#E8142A"},
    "Medium":       {"laps_peak": 22, "laps_max": 35, "pace_delta": 0.4,  "color": "#FFC800"},
    "Hard":         {"laps_peak": 35, "laps_max": 50, "pace_delta": 0.9,  "color": "#CCCCCC"},
    "Intermediate": {"laps_peak": 25, "laps_max": 40, "pace_delta": 2.0,  "color": "#39B54A"},
    "Wet":          {"laps_peak": 30, "laps_max": 50, "pace_delta": 4.0,  "color": "#0067FF"},
}

COMPOUND_COLORS = {c: v["color"] for c, v in COMPOUND_DEG.items()}

STRATEGIES = {
    1: [
        {"name": "One-Stop: M → H",  "compounds": ["Medium", "Hard"],           "risk": "Low"},
        {"name": "One-Stop: S → H",  "compounds": ["Soft", "Hard"],             "risk": "Medium"},
    ],
    2: [
        {"name": "Two-Stop: S → M → H","compounds": ["Soft","Medium","Hard"],   "risk": "Medium"},
        {"name": "Two-Stop: S → H → M","compounds": ["Soft","Hard","Medium"],   "risk": "Medium"},
        {"name": "Two-Stop: M → H → M","compounds": ["Medium","Hard","Medium"], "risk": "Low"},
    ],
    3: [
        {"name": "Aggressive: S → S → M → H","compounds":["Soft","Soft","Medium","Hard"],"risk":"High"},
        {"name": "Three-Stop: S → M → M → H","compounds":["Soft","Medium","Medium","Hard"],"risk":"High"},
    ],
}

WEATHER_ADJUSTMENTS = {
    "Dry":          {"stops_adjust": 0,  "compound_note": "Normal tyre selection applies."},
    "Cloudy":       {"stops_adjust": 0,  "compound_note": "Slightly cooler track — tyres may last a lap or two longer."},
    "Light Rain":   {"stops_adjust": 1,  "compound_note": "Consider Intermediates. Watch for dry line forming."},
    "Heavy Rain":   {"stops_adjust": 0,  "compound_note": "Full Wet tyres required. Avoid slick compounds entirely."},
}

WET_WEATHER_STRATEGIES = {
    "Light Rain": {
        1: [
            {"name": "Crossover: I → M", "compounds": ["Intermediate", "Medium"], "risk": "Medium"},
        ],
        2: [
            {"name": "Mixed: I → M → H", "compounds": ["Intermediate", "Medium", "Hard"], "risk": "Medium"},
            {"name": "Changing Track: I → I → M", "compounds": ["Intermediate", "Intermediate", "Medium"], "risk": "High"},
        ],
        3: [
            {"name": "Variable Weather: I → I → M → H", "compounds": ["Intermediate", "Intermediate", "Medium", "Hard"], "risk": "High"},
        ],
    },
    "Heavy Rain": {
        1: [
            {"name": "Wet Crossover: W → I", "compounds": ["Wet", "Intermediate"], "risk": "High"},
        ],
        2: [
            {"name": "Full Wet: W → W → W", "compounds": ["Wet", "Wet", "Wet"], "risk": "Low"},
            {"name": "Drying Race: W → I → M", "compounds": ["Wet", "Intermediate", "Medium"], "risk": "High"},
        ],
        3: [
            {"name": "Extreme Weather: W → W → I → M", "compounds": ["Wet", "Wet", "Intermediate", "Medium"], "risk": "High"},
        ],
    },
}

VALID_WEATHER = tuple(WEATHER_ADJUSTMENTS)
VALID_COMPOUNDS = tuple(COMPOUND_DEG)


def _validate_inputs(grid_position: int, weather: str, starting_compound: str) -> None:
    """Reject invalid race parameters before building a strategy."""
    if isinstance(grid_position, bool) or not isinstance(grid_position, int):
        raise TypeError("grid_position must be an integer between 1 and 20")
    if not 1 <= grid_position <= 20:
        raise ValueError("grid_position must be between 1 and 20")
    if weather not in VALID_WEATHER:
        raise ValueError(
            f"weather must be one of: {', '.join(VALID_WEATHER)}"
        )
    if starting_compound not in VALID_COMPOUNDS:
        raise ValueError(
            f"starting_compound must be one of: {', '.join(VALID_COMPOUNDS)}"
        )

CIRCUIT_STRATEGY_NOTES = {
    "monaco":       "Monaco is the ultimate undercut circuit. No real overtaking — pit before lap 25 at all costs. Track position is everything here.",
    "monza":        "Monza is a low-degradation temple of speed. One-stop on Hard tyres is optimal. Long straights mean DRS overtakes are possible.",
    "spa":          "Spa weather is unpredictable. Always have an Intermediate ready. High-speed corners cause sudden deg spikes. Eau Rouge is brutal on fronts.",
    "silverstone":  "High-energy corners destroy front-left tyres. Plan for earlier stops than the data suggests. Two-stop is safer than one.",
    "baku":         "Safety Car probability is 65%. Timing your pit around the Virtual Safety Car can gain 20+ seconds. Keep options open.",
    "marina_bay":   "Singapore is a street circuit with extremely high Safety Car probability. Night racing cools the track — tyres last longer than in day races.",
    "jeddah":       "Jeddah has the highest Safety Car risk in the calendar. Aggressive strategy pays off here — be ready to react.",
    "bahrain":      "High tyre degradation from abrasive surface. Two stops is almost always correct. First stint on Medium, then Hard is proven.",
    "hungaroring":  "The Monaco of open circuits — very difficult to overtake. Qualify well, pit early for the undercut if you fall behind.",
    "zandvoort":    "Banking corners generate high lateral loads — massive tyre deg. Two stops mandatory. Hard tyre struggles here.",
    "red_bull_ring":"Short and fast. Safety Cars are common. High tyre degradation from kerb usage. Aggressive strategy rewarded.",
    "interlagos":   "Interlagos has unpredictable weather. Soft tyres struggle on this abrasive surface. Monitor tyre temperatures carefully.",
    "default":      "Standard strategy applies. Monitor tyre degradation live and adapt to Safety Car opportunities.",
}


def _get_pit_windows(total_laps: int, n_stops: int, compounds: list[str]) -> list[dict]:
    """Calculate optimal pit windows for each stop."""
    windows = []
    if n_stops == 1:
        c1 = COMPOUND_DEG[compounds[0]]
        optimal = min(c1["laps_peak"] + 4, int(total_laps * 0.45))
        windows.append({"stop": 1, "optimal_lap": optimal,
                        "window": f"Lap {optimal-3} – {optimal+5}",
                        "from": compounds[0], "to": compounds[1]})
    elif n_stops == 2:
        c1 = COMPOUND_DEG[compounds[0]]
        c2 = COMPOUND_DEG[compounds[1]]
        s1 = min(c1["laps_peak"] + 3, int(total_laps * 0.35))
        s2 = s1 + min(c2["laps_peak"] + 3, int(total_laps * 0.38))
        s2 = min(s2, total_laps - 10)
        windows.append({"stop": 1, "optimal_lap": s1,
                        "window": f"Lap {s1-3} – {s1+4}",
                        "from": compounds[0], "to": compounds[1]})
        windows.append({"stop": 2, "optimal_lap": s2,
                        "window": f"Lap {s2-3} – {s2+4}",
                        "from": compounds[1], "to": compounds[2]})
    elif n_stops >= 3:
        stint = total_laps // (n_stops + 1)
        for i in range(n_stops):
            lap = stint * (i + 1)
            windows.append({"stop": i+1, "optimal_lap": lap,
                            "window": f"Lap {lap-2} – {lap+3}",
                            "from": compounds[i], "to": compounds[i+1]})
    return windows


def _pace_loss(
    compounds: list[str],
    windows: list[dict],
    total_laps: int,
    pit_loss_seconds: float = 22.0,
) -> float:
    """Estimate strategy time loss from pit stops, tyre pace, and degradation."""
    boundaries = [0, *[window["optimal_lap"] for window in windows], total_laps]
    total = len(windows) * pit_loss_seconds

    for index, compound in enumerate(compounds):
        stint_laps = max(0, boundaries[index + 1] - boundaries[index])
        tyre = COMPOUND_DEG.get(compound, COMPOUND_DEG["Medium"])
        total += stint_laps * tyre["pace_delta"]
        laps_over_peak = max(0, stint_laps - tyre["laps_peak"])
        total += 0.04 * laps_over_peak**2

    return round(total, 1)


def _strategy_comparison(
    primary: dict,
    alternatives: list[dict],
    total_laps: int,
    pit_loss_seconds: float,
) -> list[dict]:
    """Build comparable time-loss estimates for the recommended strategies."""
    unique_options = []
    seen = set()
    for option in [primary, *alternatives]:
        key = tuple(option["compounds"])
        if key not in seen:
            unique_options.append(option)
            seen.add(key)

    comparison = []
    for option in unique_options:
        stops = len(option["compounds"]) - 1
        windows = _get_pit_windows(total_laps, stops, option["compounds"])
        comparison.append({
            "name": option["name"],
            "compounds": option["compounds"],
            "stops": stops,
            "risk": option["risk"],
            "estimated_loss_seconds": _pace_loss(
                option["compounds"], windows, total_laps, pit_loss_seconds
            ),
            "recommended": option is primary,
        })

    fastest = min(row["estimated_loss_seconds"] for row in comparison)
    for row in comparison:
        row["delta_to_fastest_seconds"] = round(
            row["estimated_loss_seconds"] - fastest, 1
        )
    return comparison


def _undercut_opportunity(circuit_ref: str) -> str:
    char = CIRCUIT_CHARACTERISTICS.get(circuit_ref, {})
    ot = char.get("overtaking", "medium")
    if ot in ("very low", "low"):
        return "HIGH — Undercut is critical here. Pit 2–3 laps before your rival."
    elif ot == "medium":
        return "MEDIUM — Undercut is useful but overtaking is possible on track."
    else:
        return "LOW — DRS overtaking is easy. Can afford to stay out longer."


def recommend(
    circuit_name: str,
    grid_position: int,
    weather: str = "Dry",
    starting_compound: str = "Medium",
    aggressive: bool = False,
    driver_name: str = "",
) -> dict:
    """
    Full strategy recommendation.
    Returns a rich dict with primary strategy, alternatives, pit windows,
    risk ratings, safety car advice, and undercut analysis.
    """
    _validate_inputs(grid_position, weather, starting_compound)

    circuit_data = CIRCUIT_DATA_MAP.get(circuit_name, {})
    circuit_ref  = circuit_data.get("ref", "bahrain")
    total_laps   = circuit_data.get("laps", 57)
    char         = CIRCUIT_CHARACTERISTICS.get(circuit_ref, {})

    tyre_deg       = char.get("tyre_deg", "medium")
    sc_prob        = char.get("safety_car_prob", 0.40)
    overtaking     = char.get("overtaking", "medium")
    pit_loss_secs  = char.get("pit_loss", 22)

    weather_info = WEATHER_ADJUSTMENTS.get(weather, WEATHER_ADJUSTMENTS["Dry"])

    # Determine recommended stop count
    if weather in ("Heavy Rain",):
        recommended_stops = 2
    elif tyre_deg == "high":
        recommended_stops = 2 if not aggressive else 3
    elif tyre_deg == "low":
        recommended_stops = 1
    else:
        recommended_stops = 2

    # Grid position adjustment
    if grid_position > 15 and overtaking in ("very low", "low"):
        recommended_stops = min(3, recommended_stops + 1)
        grid_note = "Starting from the back at a low-overtaking circuit — aggressive strategy needed to gain positions."
    elif grid_position <= 3:
        grid_note = "Starting from the front — protect position. Conservative strategy preferred."
    elif grid_position <= 10:
        grid_note = "Starting in points positions — balance between protecting position and attacking."
    else:
        grid_note = "Starting outside top 10 — more aggressive strategy may help gain ground."

    recommended_stops += weather_info["stops_adjust"]
    recommended_stops = max(1, min(3, recommended_stops))

    # Pick primary strategy
    strategy_catalogue = WET_WEATHER_STRATEGIES.get(weather, STRATEGIES)
    options = strategy_catalogue.get(recommended_stops, strategy_catalogue[2])
    if weather not in WET_WEATHER_STRATEGIES and starting_compound in ("Soft", "Medium", "Hard"):
        # prefer options starting with user's compound
        matching = [o for o in options if o["compounds"][0] == starting_compound]
        primary = matching[0] if matching else options[0]
    elif weather in WET_WEATHER_STRATEGIES:
        safe_compounds = ("Wet", "Intermediate")
        matching = [o for o in options if o["compounds"][0] == starting_compound]
        primary = matching[0] if starting_compound in safe_compounds and matching else options[0]
    else:
        primary = options[0]

    # Alternative strategies (different stop counts and weather scenarios)
    alt_stops = 1 if recommended_stops == 2 else 2
    alternatives = [
        option
        for stop_options in strategy_catalogue.values()
        for option in stop_options
        if option is not primary
    ]
    alternatives.sort(
        key=lambda option: abs((len(option["compounds"]) - 1) - alt_stops)
    )

    # Pit windows
    pit_windows = _get_pit_windows(total_laps, recommended_stops, primary["compounds"])

    # Safety car strategy
    if sc_prob >= 0.55:
        sc_advice = f"HIGH SC probability ({sc_prob:.0%}). Leave options open — delay your planned stop by 3–5 laps so you can pit under SC for free. This can be worth 20–25 seconds."
    elif sc_prob >= 0.40:
        sc_advice = f"MODERATE SC probability ({sc_prob:.0%}). Stay alert around laps 20–40. A VSC window can save a full pit stop time loss."
    else:
        sc_advice = f"LOW SC probability ({sc_prob:.0%}). Commit to your planned windows. Reactive strategy is unlikely to be needed."

    # Undercut
    undercut = _undercut_opportunity(circuit_ref)

    # Pace loss estimate
    pace_loss = _pace_loss(
        primary["compounds"], pit_windows, total_laps, pit_loss_secs
    )
    strategy_comparison = _strategy_comparison(
        primary, alternatives, total_laps, pit_loss_secs
    )

    # Circuit note
    circuit_note_key = circuit_ref if circuit_ref in CIRCUIT_STRATEGY_NOTES else "default"
    circuit_note = CIRCUIT_STRATEGY_NOTES[circuit_note_key]
    compound_arrow = " \u2192 "
    tyre_strategy = f"{primary['name']} ({compound_arrow.join(primary['compounds'])})"

    return {
        "circuit_name":        circuit_name,
        "circuit_ref":         circuit_ref,
        "total_laps":          total_laps,
        "weather":             weather,
        "grid_position":       grid_position,
        "driver_name":         driver_name,
        "recommended_stops":   recommended_stops,
        "recommended_pit_stops": recommended_stops,
        "primary_strategy":    primary,
        "tyre_strategy":       tyre_strategy,
        "pit_windows":         pit_windows,
        "alternatives":        alternatives,
        "tyre_deg_level":      tyre_deg,
        "sc_probability":      sc_prob,
        "sc_advice":           sc_advice,
        "undercut_opportunity":undercut,
        "overtaking_ease":     overtaking,
        "pit_loss_seconds":    pit_loss_secs,
        "pace_loss_estimate":  pace_loss,
        "strategy_comparison": strategy_comparison,
        "weather_note":        weather_info["compound_note"],
        "grid_note":           grid_note,
        "circuit_note":        circuit_note,
        "starting_compound":   starting_compound,
        "compound_data":       COMPOUND_DEG,
    }
