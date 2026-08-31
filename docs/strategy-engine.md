# Strategy Engine Guide

The strategy engine combines circuit metadata, tyre degradation assumptions,
weather, grid position, and race progress. It is deterministic: identical
inputs produce an identical recommendation, which makes it suitable for the
dashboard, automated tests, and other Python clients.

## Basic recommendation

```python
from src.strategy import recommend

plan = recommend(
    circuit_name="British Grand Prix",
    grid_position=6,
    weather="Light Rain",
    starting_compound="Intermediate",
)

print(plan["tyre_strategy"])
print(plan["pit_windows"])
```

Each pit window includes `window_start`, `optimal_lap`, and `window_end` as
integers as well as a display-ready `window` label.

## Live race engineer

Pass the current lap and number of completed stops to receive an instruction
for the next planned stop:

```python
live_plan = recommend(
    circuit_name="Bahrain Grand Prix",
    grid_position=8,
    current_lap=18,
    completed_stops=0,
)

print(live_plan["live_status"]["status"])
print(live_plan["live_status"]["instruction"])
```

The status progresses through these states:

| Status | Meaning |
|---|---|
| `HOLD` | The next window is more than three laps away. |
| `PREPARE` | The next window opens within three laps. |
| `WINDOW_OPEN` | An early stop is viable, but the optimal lap is ahead. |
| `BOX` | The optimal lap has arrived or the window is about to close. |
| `OVERDUE` | The planned window was missed. |
| `COMPLETE` | All stops in the selected plan are complete. |

`current_lap` must be between zero and the circuit race distance.
`completed_stops` cannot exceed the recommendation's planned stop count.

## Important limitation

Estimated loss values compare strategies within this application. They are not
live Formula 1 telemetry predictions: fuel load, traffic, tyre temperature,
driver-specific wear, red flags, and real-time track evolution are outside the
current model.
