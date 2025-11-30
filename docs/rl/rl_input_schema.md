# RL Observation Schema

Observation construction is handled centrally by `src/polyo/rl/observation.py` and `src/polyo/rl/pipeline.py`. Base features vary slightly per environment; LSTM fields are always appended in the same order.

## Base schemas
- **GMGN RL demo (`run_rl_simulation`)**
  1. `price`
  2. `norm_price`
  3. `regime`

- **Mini demo (`apps/app_polyo.py`)**
  1. `price`
  2. `vol`
  3. `position`

- **TinyEnv (use_case_simple)**
  1. `price`
  2. `vol`
  3. `regime`
  4. `position`

- **StressEnv (use_case_stress_tests)**
  1. `price`
  2. `regime`
  3. `liquidity`
  4. `position`

## LSTM extensions
When `use_lstm=True`, the following fields are appended:
5. `lstm_next_return`
6. `lstm_vol_forecast`
7. `lstm_prob_up`

The final dimension is therefore `len(base_features)` when disabled, or `len(base_features) + 3` when enabled.

## Example vector layout
For the GMGN RL demo with LSTM enabled:

| Index | Feature              |
| ----- | -------------------- |
| 0     | price                |
| 1     | norm_price           |
| 2     | regime               |
| 3     | lstm_next_return     |
| 4     | lstm_vol_forecast    |
| 5     | lstm_prob_up         |

All builders normalise invalid values to zero, and optional normalisation of the full vector can be enabled via `ObservationBuilder(normalise=True)`.
