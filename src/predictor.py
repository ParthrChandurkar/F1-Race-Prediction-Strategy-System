"""
src/predictor.py
Future race predictor using trained models + 2024/2025 driver ratings.
Predicts Top10 probability and finishing position for any 2025 race.
"""

import os
import json
import joblib
import numpy as np
from src.f1_2024_data import DRIVER_SKILL, TEAM_CAR_RATING, DRIVER_TEAM_2025, TEAM_REFS

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")


def _load(name: str):
    return joblib.load(os.path.join(MODELS_DIR, f"{name}.pkl"))


def _scaler():
    return joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))


def _le(col: str):
    return joblib.load(os.path.join(MODELS_DIR, f"le_{col}.pkl"))


def _encode_safe(le, value: str) -> int:
    """Encode a label, returning closest known value if unseen."""
    try:
        return int(le.transform([value])[0])
    except Exception:
        # Return median class index as safe fallback
        return len(le.classes_) // 2


def build_input_vector(
    grid: int,
    qual_position: int,
    year: int,
    driver_ref: str,
    constructor_ref: str,
    circuit_ref: str,
    driver_avg_finish: float = 8.0,
    team_avg_finish: float = 8.0,
    driver_win_rate: float = 0.1,
    pit_stop_count: float = 2.0,
    avg_lap_ms: float = 90000.0,
) -> np.ndarray:
    le_driver = _le("driverRef")
    le_constructor = _le("constructorRef")
    le_circuit = _le("circuitRef")

    x = np.array([[
        grid, qual_position, year,
        _encode_safe(le_driver, driver_ref),
        _encode_safe(le_constructor, constructor_ref),
        _encode_safe(le_circuit, circuit_ref),
        driver_avg_finish, team_avg_finish,
        driver_win_rate, pit_stop_count, avg_lap_ms,
    ]], dtype=float)
    return x


def predict_driver(
    driver_name: str,
    circuit_ref: str,
    grid: int,
    qual_position: int,
    year: int = 2025,
    avg_lap_ms: float = 90000.0,
    weather_factor: float = 1.0,
) -> dict:
    """
    Full prediction for one driver at one circuit.
    Uses driver skill + team car rating to adjust model output.
    """
    team_name = DRIVER_TEAM_2025.get(driver_name, "Ferrari")
    driver_ref = driver_name.lower().replace(" ", "_").split("_")[-1]
    constructor_ref = TEAM_REFS.get(team_name, "ferrari")

    skill = DRIVER_SKILL.get(driver_name, 0.75)
    car   = TEAM_CAR_RATING.get(team_name, 0.75)
    combined = (skill * 0.6 + car * 0.4)  # driver matters more

    # Historical rolling stats estimated from skill
    driver_avg = round(10.0 - skill * 8.0, 2)  # skill 0.97 → avg ~2.2
    team_avg   = round(10.0 - car * 7.0, 2)
    win_rate   = round(max(0, skill - 0.70) * 2, 3)

    x = build_input_vector(
        grid=grid, qual_position=qual_position, year=year,
        driver_ref=driver_ref, constructor_ref=constructor_ref,
        circuit_ref=circuit_ref, driver_avg_finish=driver_avg,
        team_avg_finish=team_avg, driver_win_rate=win_rate,
        avg_lap_ms=avg_lap_ms,
    )

    sc = _scaler()
    x_s = sc.transform(x)

    rf = _load("Random_Forest")
    ridge = _load("Ridge_Regression")

    raw_prob = float(rf.predict_proba(x_s)[0][1])
    raw_pos  = float(ridge.predict(x_s)[0])

    # Adjust with real-world driver+car rating
    # Add circuit-specific randomness so predictions vary per circuit
    import hashlib
    seed_hash = int(hashlib.md5(f"{driver_name}{circuit_ref}{year}".encode()).hexdigest()[:6], 16)
    rng2 = np.random.default_rng(seed_hash % (2**31))
    circuit_noise = float(rng2.normal(0, 0.05))  # ±5% variation per driver/circuit combo

    adj_prob = np.clip(raw_prob * 0.5 + combined * 0.5 + circuit_noise, 0.01, 0.99)
    adj_pos  = np.clip(raw_pos * 0.5 + (1 - combined) * 19 * 0.5 + 1 - circuit_noise * 3, 1.0, 20.0)

    adj_prob = float(adj_prob) * weather_factor
    adj_prob = np.clip(adj_prob, 0.01, 0.99)

    return {
        "driver":      driver_name,
        "team":        team_name,
        "top10_prob":  round(float(adj_prob), 4),
        "win_prob":    round(float(adj_prob) * combined * 0.3, 4),
        "podium_prob": round(float(adj_prob) * combined * 0.6, 4),
        "pred_position": round(float(adj_pos), 1),
        "skill_rating": skill,
        "car_rating":   car,
    }


def predict_full_grid(
    circuit_ref: str,
    circuit_avg_lap_ms: float,
    year: int = 2025,
    weather_factor: float = 1.0,
) -> list[dict]:
    """
    Predict all 20 drivers for a race.
    Assigns grid/qual positions based on team/driver ratings with noise.
    Returns list sorted by predicted position.
    """
    from src.f1_2024_data import DRIVER_NAMES_2024
    import random
    # Use circuit_ref + weather as seed basis so results vary per circuit/condition
    import hashlib
    seed_str = f"{circuit_ref}_{weather_factor:.2f}_{year}"
    seed_val = int(hashlib.md5(seed_str.encode()).hexdigest()[:8], 16) % (2**31)
    rng = np.random.default_rng(seed_val)

    # Circuit-specific performance adjustments
    # Some drivers genuinely perform better/worse at specific circuits
    circuit_bonus = {
        "monaco":       {"hamilton":0.06,"alonso":0.05,"leclerc":0.04,"verstappen":0.02},
        "monza":        {"norris":0.05,"piastri":0.04,"sainz":0.03},
        "spa":          {"verstappen":0.06,"hamilton":0.04,"norris":0.03},
        "silverstone":  {"hamilton":0.08,"norris":0.06,"russell":0.05},
        "red_bull_ring":{"verstappen":0.07,"perez":0.04},
        "hungaroring":  {"hamilton":0.05,"russell":0.04,"alonso":0.03},
        "suzuka":       {"verstappen":0.06,"norris":0.04},
        "interlagos":   {"hamilton":0.06,"verstappen":0.04,"norris":0.03},
        "bahrain":      {"leclerc":0.04,"hamilton":0.03},
        "albert_park":  {"leclerc":0.05,"sainz":0.04,"norris":0.03},
        "baku":         {"perez":0.06,"leclerc":0.04,"sainz":0.05},
        "marina_bay":   {"alonso":0.04,"hamilton":0.03},
        "zandvoort":    {"verstappen":0.09,"norris":0.04},
    }
    this_circuit_bonus = circuit_bonus.get(circuit_ref, {})

    # Generate qualifying order based on ratings + circuit bonus + noise
    qual_scores = []
    for drv in DRIVER_NAMES_2024:
        skill = DRIVER_SKILL.get(drv, 0.72)
        car   = TEAM_CAR_RATING.get(DRIVER_TEAM_2025.get(drv, "Haas"), 0.65)
        dref  = drv.lower().split()[-1]
        bonus = this_circuit_bonus.get(dref, 0.0)
        # noise: 0.06 std gives meaningful lap-by-lap variation (~1-2 positions)
        score = (skill * 0.55 + car * 0.45) + bonus + rng.normal(0, 0.06)
        qual_scores.append((drv, score))

    qual_scores.sort(key=lambda x: x[1], reverse=True)
    qual_order = {drv: i+1 for i, (drv, _) in enumerate(qual_scores)}

    results = []
    for drv in DRIVER_NAMES_2024:
        qpos  = qual_order[drv]
        grid  = qpos  # assuming no grid penalties for simulation
        res   = predict_driver(drv, circuit_ref, grid, qpos, year,
                               circuit_avg_lap_ms, weather_factor)
        res["grid_position"] = grid
        res["qual_position"] = qpos
        results.append(res)

    results.sort(key=lambda x: x["pred_position"])
    for i, r in enumerate(results):
        r["predicted_finish"] = i + 1

    return results


def load_metrics() -> dict:
    with open(os.path.join(MODELS_DIR, "metrics.json")) as f:
        return json.load(f)


def load_feature_importance() -> dict:
    with open(os.path.join(MODELS_DIR, "feature_importance.json")) as f:
        return json.load(f)


def load_meta() -> dict:
    with open(os.path.join(MODELS_DIR, "meta.json")) as f:
        return json.load(f)
