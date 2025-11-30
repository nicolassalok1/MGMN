import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow import keras

from technical_analysis.coppock import Coppock
from technical_analysis.dpo import Dpo
from technical_analysis.generate_labels import Genlabels
from technical_analysis.macd import Macd
from technical_analysis.poly_interpolation import PolyInter
from technical_analysis.rsi import StochRsi

ROOT = Path(__file__).resolve().parent


def extract_data(data, progress=False):
    labels = Genlabels(data, window=25, polyorder=3).labels

    macd = Macd(data, 6, 12, 3).values
    stoch_rsi = StochRsi(data, period=14).hist_values
    dpo = Dpo(data, period=4).values
    cop = Coppock(data, wma_pd=10, roc_long=6, roc_short=3).values
    inter_slope = PolyInter(data, progress_bar=progress).values

    X = np.array(
        [
            macd[30:-1],
            stoch_rsi[30:-1],
            inter_slope[30:-1],
            dpo[30:-1],
            cop[30:-1],
        ]
    )

    X = np.transpose(X)
    labels = labels[31:]

    return X, labels


def adjust_data(X, y, split=0.8):
    count_1 = np.count_nonzero(y)
    count_0 = y.shape[0] - count_1
    cut = min(count_0, count_1)

    train_idx = int(cut * split)

    np.random.seed(42)
    shuffle_index = np.random.permutation(X.shape[0])
    X, y = X[shuffle_index], y[shuffle_index]

    idx_1 = np.argwhere(y == 1).flatten()
    idx_0 = np.argwhere(y == 0).flatten()

    X_train = np.concatenate((X[idx_1[:train_idx]], X[idx_0[:train_idx]]), axis=0)
    X_test = np.concatenate((X[idx_1[train_idx:cut]], X[idx_0[train_idx:cut]]), axis=0)
    y_train = np.concatenate((y[idx_1[:train_idx]], y[idx_0[:train_idx]]), axis=0)
    y_test = np.concatenate((y[idx_1[train_idx:cut]], y[idx_0[train_idx:cut]]), axis=0)

    np.random.seed(7)
    shuffle_train = np.random.permutation(X_train.shape[0])
    shuffle_test = np.random.permutation(X_test.shape[0])

    X_train, y_train = X_train[shuffle_train], y_train[shuffle_train]
    X_test, y_test = X_test[shuffle_test], y_test[shuffle_test]

    return X_train, X_test, y_train, y_test


def shape_data(X, y, timesteps=10, model_dir=None, save_scaler=True):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    if model_dir is None:
        model_dir = ROOT / "models"
    model_dir = Path(model_dir)
    if save_scaler:
        model_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(scaler, model_dir / "scaler.dump")

    reshaped = []
    for i in range(timesteps, X_scaled.shape[0] + 1):
        reshaped.append(X_scaled[i - timesteps : i])

    X_seq = np.array(reshaped)
    y_seq = y[timesteps - 1 :]

    return X_seq, y_seq


def build_model(input_shape, lstm_units=32, dense_units=16, dropout=0.2):
    model = keras.models.Sequential()
    model.add(keras.Input(shape=input_shape))
    model.add(keras.layers.LSTM(lstm_units, return_sequences=True))
    model.add(keras.layers.Dropout(dropout))

    model.add(keras.layers.LSTM(lstm_units, return_sequences=False))
    model.add(keras.layers.Dropout(dropout))

    model.add(keras.layers.Dense(dense_units, activation="relu"))
    model.add(keras.layers.Dense(2, activation="softmax"))

    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model


def load_price_history(path=None):
    target = Path(path) if path else ROOT / "historical_data" / "hist_data.json"
    with open(target) as f:
        data = json.load(f)
    return np.array(data["close"], dtype=float)


def train_model(
    prices,
    timesteps=10,
    epochs=10,
    batch_size=8,
    model_dir=None,
    progress=False,
    verbose=2,
):
    if model_dir is None:
        model_dir = ROOT / "models"
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    X_raw, y_raw = extract_data(np.array(prices, dtype=float), progress=progress)
    X_seq, y_seq = shape_data(X_raw, y_raw, timesteps=timesteps, model_dir=model_dir)
    X_train, X_test, y_train, y_test = adjust_data(X_seq, y_seq)

    y_train = keras.utils.to_categorical(y_train, 2)
    y_test = keras.utils.to_categorical(y_test, 2)

    model = build_model((X_train.shape[1], X_train.shape[2]))
    history = model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        shuffle=True,
        validation_data=(X_test, y_test),
        verbose=verbose,
    )
    # Save in native Keras format to avoid legacy HDF5 warning
    model.save(model_dir / "lstm_model.keras")
    return model, history, (X_test, y_test)


if __name__ == "__main__":
    price_history = load_price_history()
    train_model(price_history, progress=True)
