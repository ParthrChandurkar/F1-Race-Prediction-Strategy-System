"""
src/f1_2024_data.py
Hardcoded 2024 F1 season data — 20 drivers, 10 teams, 24 circuits.
Used for future race prediction inputs.
"""

# 2024 F1 Driver grid — name: {team, number, nationality, driverRef}
DRIVERS_2024 = {
    "Max Verstappen":      {"team": "Red Bull Racing",  "number": 1,  "nationality": "Dutch",     "ref": "verstappen"},
    "Sergio Perez":        {"team": "Red Bull Racing",  "number": 11, "nationality": "Mexican",   "ref": "perez"},
    "Lewis Hamilton":      {"team": "Ferrari",          "number": 44, "nationality": "British",   "ref": "hamilton"},
    "Charles Leclerc":     {"team": "Ferrari",          "number": 16, "nationality": "Monegasque","ref": "leclerc"},
    "Lando Norris":        {"team": "McLaren",          "number": 4,  "nationality": "British",   "ref": "norris"},
    "Oscar Piastri":       {"team": "McLaren",          "number": 81, "nationality": "Australian","ref": "piastri"},
    "George Russell":      {"team": "Mercedes",         "number": 63, "nationality": "British",   "ref": "russell"},
    "Kimi Antonelli":      {"team": "Mercedes",         "number": 12, "nationality": "Italian",   "ref": "antonelli"},
    "Fernando Alonso":     {"team": "Aston Martin",     "number": 14, "nationality": "Spanish",   "ref": "alonso"},
    "Lance Stroll":        {"team": "Aston Martin",     "number": 18, "nationality": "Canadian",  "ref": "stroll"},
    "Nico Hulkenberg":     {"team": "Haas",             "number": 27, "nationality": "German",    "ref": "hulkenberg"},
    "Esteban Ocon":        {"team": "Haas",             "number": 31, "nationality": "French",    "ref": "ocon"},
    "Pierre Gasly":        {"team": "Alpine",           "number": 10, "nationality": "French",    "ref": "gasly"},
    "Jack Doohan":         {"team": "Alpine",           "number": 7,  "nationality": "Australian","ref": "doohan"},
    "Alexander Albon":     {"team": "Williams",         "number": 23, "nationality": "Thai",      "ref": "albon"},
    "Carlos Sainz":        {"team": "Williams",         "number": 55, "nationality": "Spanish",   "ref": "sainz"},
    "Liam Lawson":         {"team": "Racing Bulls",     "number": 30, "nationality": "New Zealander","ref": "lawson"},
    "Yuki Tsunoda":        {"team": "Racing Bulls",     "number": 22, "nationality": "Japanese",  "ref": "tsunoda"},
    "Valtteri Bottas":     {"team": "Kick Sauber",      "number": 77, "nationality": "Finnish",   "ref": "bottas"},
    "Nico Hulkenberg":     {"team": "Kick Sauber",      "number": 27, "nationality": "German",    "ref": "hulkenberg"},
}

# Clean deduplicated list
DRIVER_NAMES_2024 = [
    "Max Verstappen", "Sergio Perez", "Lewis Hamilton", "Charles Leclerc",
    "Lando Norris", "Oscar Piastri", "George Russell", "Kimi Antonelli",
    "Fernando Alonso", "Lance Stroll", "Nico Hulkenberg", "Esteban Ocon",
    "Pierre Gasly", "Jack Doohan", "Alexander Albon", "Carlos Sainz",
    "Liam Lawson", "Yuki Tsunoda", "Valtteri Bottas", "Zhou Guanyu",
]

TEAMS_2024 = [
    "Red Bull Racing", "Ferrari", "McLaren", "Mercedes",
    "Aston Martin", "Alpine", "Williams", "Racing Bulls",
    "Haas", "Kick Sauber",
]

TEAM_REFS = {
    "Red Bull Racing": "red_bull",
    "Ferrari":         "ferrari",
    "McLaren":         "mclaren",
    "Mercedes":        "mercedes",
    "Aston Martin":    "aston_martin",
    "Alpine":          "alpine",
    "Williams":        "williams",
    "Racing Bulls":    "alphatauri",
    "Haas":            "haas",
    "Kick Sauber":     "sauber",
}

DRIVER_TEAM_2025 = {
    "Max Verstappen": "Red Bull Racing",
    "Sergio Perez":   "Red Bull Racing",
    "Lewis Hamilton": "Ferrari",
    "Charles Leclerc":"Ferrari",
    "Lando Norris":   "McLaren",
    "Oscar Piastri":  "McLaren",
    "George Russell": "Mercedes",
    "Kimi Antonelli": "Mercedes",
    "Fernando Alonso":"Aston Martin",
    "Lance Stroll":   "Aston Martin",
    "Nico Hulkenberg":"Haas",
    "Esteban Ocon":   "Haas",
    "Pierre Gasly":   "Alpine",
    "Jack Doohan":    "Alpine",
    "Alexander Albon":"Williams",
    "Carlos Sainz":   "Williams",
    "Liam Lawson":    "Racing Bulls",
    "Yuki Tsunoda":   "Racing Bulls",
    "Valtteri Bottas":"Kick Sauber",
    "Zhou Guanyu":    "Kick Sauber",
}

# 2025 race calendar circuits with refs matching dataset
CIRCUITS_2025 = [
    {"name": "Bahrain Grand Prix",          "ref": "bahrain",      "country": "Bahrain",      "laps": 57, "lap_ms": 95000},
    {"name": "Saudi Arabian Grand Prix",    "ref": "jeddah",       "country": "Saudi Arabia", "laps": 50, "lap_ms": 88000},
    {"name": "Australian Grand Prix",       "ref": "albert_park",  "country": "Australia",    "laps": 58, "lap_ms": 83000},
    {"name": "Japanese Grand Prix",         "ref": "suzuka",       "country": "Japan",        "laps": 53, "lap_ms": 93000},
    {"name": "Chinese Grand Prix",          "ref": "shanghai",     "country": "China",        "laps": 56, "lap_ms": 96000},
    {"name": "Miami Grand Prix",            "ref": "miami",        "country": "USA",          "laps": 57, "lap_ms": 92000},
    {"name": "Emilia Romagna Grand Prix",   "ref": "imola",        "country": "Italy",        "laps": 63, "lap_ms": 78000},
    {"name": "Monaco Grand Prix",           "ref": "monaco",       "country": "Monaco",       "laps": 78, "lap_ms": 74000},
    {"name": "Canadian Grand Prix",         "ref": "villeneuve",   "country": "Canada",       "laps": 70, "lap_ms": 76000},
    {"name": "Spanish Grand Prix",          "ref": "catalunya",    "country": "Spain",        "laps": 66, "lap_ms": 82000},
    {"name": "Austrian Grand Prix",         "ref": "red_bull_ring","country": "Austria",      "laps": 71, "lap_ms": 68000},
    {"name": "British Grand Prix",          "ref": "silverstone",  "country": "UK",           "laps": 52, "lap_ms": 87000},
    {"name": "Belgian Grand Prix",          "ref": "spa",          "country": "Belgium",      "laps": 44, "lap_ms": 107000},
    {"name": "Hungarian Grand Prix",        "ref": "hungaroring",  "country": "Hungary",      "laps": 70, "lap_ms": 79000},
    {"name": "Dutch Grand Prix",            "ref": "zandvoort",    "country": "Netherlands",  "laps": 72, "lap_ms": 74000},
    {"name": "Italian Grand Prix",          "ref": "monza",        "country": "Italy",        "laps": 53, "lap_ms": 84000},
    {"name": "Azerbaijan Grand Prix",       "ref": "baku",         "country": "Azerbaijan",   "laps": 51, "lap_ms": 103000},
    {"name": "Singapore Grand Prix",        "ref": "marina_bay",   "country": "Singapore",    "laps": 62, "lap_ms": 100000},
    {"name": "United States Grand Prix",    "ref": "americas",     "country": "USA",          "laps": 56, "lap_ms": 98000},
    {"name": "Mexico City Grand Prix",      "ref": "rodriguez",    "country": "Mexico",       "laps": 71, "lap_ms": 79000},
    {"name": "São Paulo Grand Prix",        "ref": "interlagos",   "country": "Brazil",       "laps": 71, "lap_ms": 74000},
    {"name": "Las Vegas Grand Prix",        "ref": "vegas",        "country": "USA",          "laps": 50, "lap_ms": 94000},
    {"name": "Qatar Grand Prix",            "ref": "losail",       "country": "Qatar",        "laps": 57, "lap_ms": 84000},
    {"name": "Abu Dhabi Grand Prix",        "ref": "yas_marina",   "country": "UAE",          "laps": 58, "lap_ms": 93000},
]

CIRCUIT_NAMES_2025 = [c["name"] for c in CIRCUITS_2025]
CIRCUIT_REF_MAP   = {c["name"]: c["ref"] for c in CIRCUITS_2025}
CIRCUIT_DATA_MAP  = {c["name"]: c for c in CIRCUITS_2025}

# Driver skill ratings (0-1, based on 2024 season performance)
DRIVER_SKILL = {
    "Max Verstappen": 0.97, "Lewis Hamilton": 0.91, "Charles Leclerc": 0.88,
    "Lando Norris":   0.90, "Oscar Piastri":  0.85, "George Russell":  0.84,
    "Fernando Alonso":0.86, "Carlos Sainz":   0.83, "Sergio Perez":    0.78,
    "Kimi Antonelli": 0.72, "Lance Stroll":   0.70, "Yuki Tsunoda":    0.73,
    "Nico Hulkenberg":0.76, "Pierre Gasly":   0.74, "Alexander Albon": 0.75,
    "Esteban Ocon":   0.71, "Liam Lawson":    0.70, "Valtteri Bottas": 0.72,
    "Jack Doohan":    0.65, "Zhou Guanyu":    0.68,
}

# Dataset driverRef lookup — matches actual Kaggle CSV driverRef column
DRIVER_DATASET_REF = {
    "Max Verstappen":  "max_verstappen",
    "Lewis Hamilton":  "hamilton",
    "Charles Leclerc": "leclerc",
    "Lando Norris":    "norris",
    "Oscar Piastri":   "piastri",
    "George Russell":  "russell",
    "Fernando Alonso": "alonso",
    "Carlos Sainz":    "sainz",
    "Sergio Perez":    "perez",
    "Lance Stroll":    "stroll",
    "Valtteri Bottas": "bottas",
    "Yuki Tsunoda":    "tsunoda",
    "Pierre Gasly":    "gasly",
    "Alexander Albon": "albon",
    "Esteban Ocon":    "ocon",
    "Nico Hulkenberg": "hulkenberg",
    "Kimi Antonelli":  "antonelli",
    "Jack Doohan":     "doohan",
    "Liam Lawson":     "lawson",
    "Zhou Guanyu":     "zhou",
}

# Team car performance ratings (0-1)
TEAM_CAR_RATING = {
    "Red Bull Racing": 0.95, "McLaren": 0.92, "Ferrari": 0.90,
    "Mercedes": 0.87,        "Aston Martin": 0.78, "Alpine": 0.68,
    "Williams": 0.70,        "Racing Bulls": 0.72, "Haas": 0.65,
    "Kick Sauber": 0.60,
}

# Circuit characteristics for strategy
CIRCUIT_CHARACTERISTICS = {
    "bahrain":       {"overtaking": "medium", "tyre_deg": "high",   "safety_car_prob": 0.35, "pit_loss": 22},
    "jeddah":        {"overtaking": "high",   "tyre_deg": "medium", "safety_car_prob": 0.55, "pit_loss": 20},
    "albert_park":   {"overtaking": "medium", "tyre_deg": "medium", "safety_car_prob": 0.45, "pit_loss": 24},
    "suzuka":        {"overtaking": "low",    "tyre_deg": "high",   "safety_car_prob": 0.30, "pit_loss": 26},
    "shanghai":      {"overtaking": "medium", "tyre_deg": "medium", "safety_car_prob": 0.40, "pit_loss": 25},
    "miami":         {"overtaking": "medium", "tyre_deg": "high",   "safety_car_prob": 0.50, "pit_loss": 23},
    "imola":         {"overtaking": "low",    "tyre_deg": "medium", "safety_car_prob": 0.35, "pit_loss": 27},
    "monaco":        {"overtaking": "very low","tyre_deg": "low",   "safety_car_prob": 0.70, "pit_loss": 18},
    "villeneuve":    {"overtaking": "medium", "tyre_deg": "low",    "safety_car_prob": 0.55, "pit_loss": 22},
    "catalunya":     {"overtaking": "low",    "tyre_deg": "high",   "safety_car_prob": 0.30, "pit_loss": 24},
    "red_bull_ring": {"overtaking": "high",   "tyre_deg": "high",   "safety_car_prob": 0.40, "pit_loss": 21},
    "silverstone":   {"overtaking": "medium", "tyre_deg": "high",   "safety_car_prob": 0.35, "pit_loss": 25},
    "spa":           {"overtaking": "high",   "tyre_deg": "medium", "safety_car_prob": 0.45, "pit_loss": 30},
    "hungaroring":   {"overtaking": "low",    "tyre_deg": "high",   "safety_car_prob": 0.30, "pit_loss": 24},
    "zandvoort":     {"overtaking": "low",    "tyre_deg": "high",   "safety_car_prob": 0.35, "pit_loss": 23},
    "monza":         {"overtaking": "high",   "tyre_deg": "low",    "safety_car_prob": 0.40, "pit_loss": 25},
    "baku":          {"overtaking": "high",   "tyre_deg": "low",    "safety_car_prob": 0.65, "pit_loss": 20},
    "marina_bay":    {"overtaking": "low",    "tyre_deg": "medium", "safety_car_prob": 0.60, "pit_loss": 22},
    "americas":      {"overtaking": "medium", "tyre_deg": "high",   "safety_car_prob": 0.45, "pit_loss": 25},
    "rodriguez":     {"overtaking": "medium", "tyre_deg": "low",    "safety_car_prob": 0.35, "pit_loss": 24},
    "interlagos":    {"overtaking": "high",   "tyre_deg": "medium", "safety_car_prob": 0.55, "pit_loss": 23},
    "vegas":         {"overtaking": "high",   "tyre_deg": "low",    "safety_car_prob": 0.45, "pit_loss": 22},
    "losail":        {"overtaking": "medium", "tyre_deg": "high",   "safety_car_prob": 0.35, "pit_loss": 23},
    "yas_marina":    {"overtaking": "medium", "tyre_deg": "medium", "safety_car_prob": 0.30, "pit_loss": 24},
}
