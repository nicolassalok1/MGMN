import shutil
import tempfile
from pathlib import Path

import numpy as np
from tensorflow import keras

from lstm import adjust_data, build_model, extract_data, load_price_history, shape_data


def main():
    prices = load_price_history()

    artifacts = Path(tempfile.mkdtemp(prefix="lstm_artifacts_"))
    try:
        X_raw, labels = extract_data(prices, progress=False)
        X_seq, labels_seq = shape_data(X_raw, labels, timesteps=10, model_dir=artifacts, save_scaler=True)

        if X_seq.ndim != 3 or X_seq.shape[0] == 0:
            raise RuntimeError("Sequence shaping failed; no samples available.")

        X_train, X_test, y_train, y_test = adjust_data(X_seq, labels_seq)
        if X_test.shape[0] == 0:
            raise RuntimeError("Test split has no samples; cannot evaluate model.")

        y_train = keras.utils.to_categorical(y_train, 2)
        y_test = keras.utils.to_categorical(y_test, 2)

        model = build_model((X_train.shape[1], X_train.shape[2]))
        model.fit(
            X_train,
            y_train,
            epochs=1,
            batch_size=8,
            shuffle=True,
            validation_data=(X_test, y_test),
            verbose=0,
        )
        loss, acc = model.evaluate(X_test, y_test, verbose=0)

        sample = min(2, X_test.shape[0])
        preds = model.predict(X_test[:sample], verbose=0)
        if preds.shape != (sample, 2):
            raise RuntimeError(f"Unexpected prediction shape: {preds.shape}")
        if not np.allclose(preds.sum(axis=1), 1, atol=1e-5):
            raise RuntimeError("Prediction probabilities do not sum to 1.")

        model_path = artifacts / "lstm_model.keras"
        model.save(model_path)

        print(f"lstm self-test ok loss={loss:.4f} acc={acc:.4f} saved={model_path}")
    finally:
        shutil.rmtree(artifacts, ignore_errors=True)


if __name__ == "__main__":
    main()
