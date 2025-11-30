# LSTM Artifacts

This folder stores lightweight mock artifacts for the POLYO LSTM forecaster.

- `model.pt` – optional torch checkpoint (created by `python -m polyo.lstm_forecaster.train`).
- `scaler.pkl` – placeholder for any preprocessing scaler.
- `meta.json` – metadata describing the mock training run.

The code gracefully falls back to heuristic forecasts if these files are missing.
