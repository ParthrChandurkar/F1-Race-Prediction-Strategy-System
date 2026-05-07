"""
src/simulator.py
Monte Carlo race simulation using predicted probabilities + driver/car ratings.
"""

import numpy as np
from src.f1_2024_data import DRIVER_SKILL, TEAM_CAR_RATING, DRIVER_TEAM_2025


def run_simulation(
    drivers: list[str],
    top10_probs: list[float],
    n_sims: int = 1000,
    seed: int = 42,
    circuit_overtaking: str = "medium",
) -> dict:
    """
    Monte Carlo simulation.
    overtaking affects how much grid position locks in the result.
    """
    rng  = np.random.default_rng(seed)
    n    = len(drivers)
    probs = np.clip(np.array(top10_probs, dtype=float), 0.01, 0.99)

    # Overtaking factor: low overtaking → grid position matters more
    ot_factor = {"very low": 0.7, "low": 0.5, "medium": 0.3, "high": 0.15}.get(circuit_overtaking, 0.3)

    win_counts    = np.zeros(n)
    podium_counts = np.zeros(n)
    top10_counts  = np.zeros(n)
    pos_sum       = np.zeros(n)
    dnf_counts    = np.zeros(n)

    for _ in range(n_sims):
        # DNF chance (~15% per race across field)
        dnf_mask = rng.random(n) < 0.075

        # Base score from model probability + noise + overtaking lock-in
        qual_scores = np.arange(n, 0, -1) / n  # front = 1, back = 0
        model_scores = probs + rng.normal(0, 0.12, n)
        # blend: high ot_factor = grid matters more
        scores = (1 - ot_factor) * model_scores + ot_factor * qual_scores
        scores[dnf_mask] = -1.0  # DNFs go to back

        order = np.argsort(scores)[::-1]

        for finish_pos, driver_idx in enumerate(order):
            pos = finish_pos + 1
            pos_sum[driver_idx] += pos
            if dnf_mask[driver_idx]:
                dnf_counts[driver_idx] += 1
            if pos == 1:
                win_counts[driver_idx] += 1
            if pos <= 3:
                podium_counts[driver_idx] += 1
            if pos <= 10:
                top10_counts[driver_idx] += 1

    results = []
    for i, drv in enumerate(drivers):
        team = DRIVER_TEAM_2025.get(drv, "")
        results.append({
            "driver":       drv,
            "team":         team,
            "win_prob":     round(win_counts[i] / n_sims, 4),
            "podium_prob":  round(podium_counts[i] / n_sims, 4),
            "top10_prob":   round(top10_counts[i] / n_sims, 4),
            "dnf_prob":     round(dnf_counts[i] / n_sims, 4),
            "avg_finish":   round(float(pos_sum[i]) / n_sims, 2),
        })

    results.sort(key=lambda r: r["avg_finish"])
    for rank, r in enumerate(results, 1):
        r["sim_position"] = rank

    return {
        "n_sims":       n_sims,
        "results":      results,
        "winner":       results[0]["driver"],
        "winner_team":  results[0]["team"],
        "top3":         [r["driver"] for r in results[:3]],
    }
