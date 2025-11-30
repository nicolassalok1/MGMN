# LSTM Integration Architecture

The LSTM forecaster is a first-class component inside the lightweight POLYO decision loop. It is optional and fully backward compatible.

## Placement in the pipeline
- **Feature prep**: past prices (and optionally volumes) are sliced per step via `build_feature_window` (`src/polyo/lstm_forecaster/utils.py`).
- **Forecasting**: `LSTMForecaster.predict` consumes the feature window and returns a dict with `next_return`, `vol_forecast`, and `prob_up`. If torch artifacts are unavailable, a heuristic fallback keeps the pipeline running.
- **Observation build (single source of truth)**: `build_obs_with_lstm` (`src/polyo/rl/pipeline.py`) calls the forecaster and appends its outputs to the base RL observation.
- **Consumers**: `TinyEnv`, `StressEnv`, `MiniEnv`, and `run_rl_simulation` all rely on `build_obs_with_lstm`, so every decision step sees the augmented observation when `use_lstm=True`.

## Data flow
1. Prices arrive (synthetic rough paths or GMGN API).
2. Per step, prices up to `t` are fed to `build_feature_window` with the configured features/horizon.
3. LSTM forecasts are produced (or skipped when disabled/unavailable).
4. Base features (price, vol/regime/position depending on the environment) are concatenated with LSTM outputs into `obs`.
5. The RL policy/logic uses `obs`; the replay/history retains the expanded vector for analysis.

## Configuration knobs
- `PolyoConfig` fields:
  - `use_lstm` (bool)
  - `lstm_horizon` (int)
  - `lstm_features` (list of feature names for the window builder)
  - `lstm_model_path` (checkpoint path)
  - `lstm_config_path` (YAML path; defaults to `config/lstm.yaml`)
- Streamlit surfaces these in the sidebar (`app_gmgn_polyo.py`, `apps/app_polyo.py`).

## Backward compatibility
- When `use_lstm=False`, `build_obs_with_lstm` skips forecasting and returns the original observation shape.
- Missing model files do not raise; the heuristic path is used so UI demos and tests stay green even without heavy dependencies.
