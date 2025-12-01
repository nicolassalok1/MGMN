import numpy as np
import pandas as pd


class StateBuilder:
    def __init__(self, window: int = 20, feature_cols: list[str] | None = None):
        self.window = window
        self.feature_cols = feature_cols

    def fit(self, df: pd.DataFrame):
        if self.feature_cols is None:
            self.feature_cols = [
                c for c in df.columns if c not in ("price", "step", "path")
            ]
        return self

    @property
    def state_dim(self) -> int:
        if not self.feature_cols:
            return 0
        return len(self.feature_cols) * self.window

    def build_state(self, df: pd.DataFrame, idx: int) -> np.ndarray:
        """
        Returns a flattened window of features ending at idx (inclusive).
        """
        start = max(0, idx - self.window + 1)
        window_df = df.iloc[start : idx + 1]
        padded = window_df
        if len(window_df) < self.window:
            pad_length = self.window - len(window_df)
            pad_block = pd.DataFrame(
                np.zeros((pad_length, len(self.feature_cols))), columns=self.feature_cols
            )
            padded = pd.concat([pad_block, window_df], ignore_index=True)
        values = padded[self.feature_cols].to_numpy(dtype=np.float32)
        return values.flatten()
