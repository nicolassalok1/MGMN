# POLYO Workspace with LSTM Forecasting

This repository glues together lightweight quant components (rough volatility, jump diffusion, Kalman smoothing, HMM regimes) with a toy RL pipeline. An optional LSTM forecaster can now extend the RL observation vector with forward-looking features.

## What changed
- New package under `src/polyo` housing a configurable `PolyoConfig`, the `LSTMForecaster`, and a single-source observation builder (`build_obs_with_lstm`).
- Streamlit apps (`app_gmgn_polyo.py`, `apps/app_polyo.py`) expose LSTM toggles and feed LSTM forecasts into the RL demo inputs.
- Tests and docs document the expanded RL input schema and LSTM usage.

## Workspace / IDE
- Use VS Code and open the workspace `MGMN.code-workspace`.
- Launch VS Code from a `Visual Studio 2022 Developer PowerShell v17.14.21` session once the PowerShell install phase is complete (banner: Copyright (c) 2025 Microsoft Corporation) so PATH and toolchain settings are preloaded in the integrated terminal.

## Quickstart
1. From a `Visual Studio 2022 Developer PowerShell v17.14.21`, run the one-shot setup (creates `polyo-gpu`, installs PyTorch, TF 2.18 + JAX 0.4.28 with `ml-dtypes==0.4.0`, editable packages, and builds limit-order-book when MSVC is available):
   ```pwsh
   pwsh -NoProfile -ExecutionPolicy Bypass -File .\set_me.ps1
   ```
2. Install dependencies (conda/venv) and ensure `src` is on `PYTHONPATH` (the script already exports `PYTHONPATH` for the session).
2. Train or refresh mock LSTM artifacts (optional, lightweight):
   ```bash
   python -m polyo.lstm_forecaster.train --config config/lstm.yaml
   ```
3. Run the Streamlit playground:
   ```bash
   streamlit run app_gmgn_polyo.py
   ```
   Use the sidebar to toggle `Activer LSTM`, set the horizon, select features, and point to a model file.

## Enabling / disabling LSTM
- Toggle `USE_LSTM` via the sidebar in the Streamlit apps or by setting `PolyoConfig(use_lstm=True/False)`.
- When enabled, the RL observation vector is augmented with:
  - `lstm_next_return`
  - `lstm_vol_forecast`
  - `lstm_prob_up`
- When disabled, the observation shape reverts to its original size and no LSTM calls are made.

## Training the forecaster
- Edit `config/lstm.yaml` to adjust hyperparameters, features, and horizon.
- Run `python -m polyo.lstm_forecaster.train --config config/lstm.yaml` to write artifacts under `models/lstm/` (`model.pt`, `scaler.pkl`, `meta.json`).
- Inference helper: `python -m polyo.lstm_forecaster.predict --config config/lstm.yaml --horizon 8`.

## RL input schema
- Observation construction is centralised in `src/polyo/rl/observation.py` and `src/polyo/rl/pipeline.py`.
- Base features per demo:
  - GMGN RL demo: `price`, `norm_price`, `regime`
  - Mini demo (`apps/app_polyo.py`): `price`, `vol`, `position`
  - Stress/simple pipelines: e.g., `price`, `vol`, `regime`, `position` (see docs for full lists)
- LSTM fields are appended in the fixed order above to avoid shape mismatches.

## Backtesting / simulation with LSTM
- `build_obs_with_lstm` is used by all toy environments (`TinyEnv`, `StressEnv`, `MiniEnv`, `run_rl_simulation`). Only past prices are fed to the LSTM, preventing leakage during replay/backtests.
- For custom simulations, pass a `PolyoConfig` and optional cached forecaster to `build_obs_with_lstm` to keep observation shapes consistent.
