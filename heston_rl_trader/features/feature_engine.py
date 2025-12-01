import numpy as np
import pandas as pd


class FeatureEngineer:
    def __init__(self, return_window: int = 10, vol_window: int = 20):
        self.return_window = return_window
        self.vol_window = vol_window

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Expects columns: price, variance (optional). Produces
        log_return, rolling_mean, rolling_vol, price_zscore.
        """
        data = df.copy()
        safe_price = data["price"].clip(lower=1e-12)
        data["log_return"] = np.log(safe_price).diff()
        data["log_return"].fillna(0.0, inplace=True)

        data["rolling_mean"] = (
            data["log_return"]
            .rolling(self.return_window, min_periods=1)
            .mean()
        )
        data["rolling_vol"] = (
            data["log_return"]
            .rolling(self.vol_window, min_periods=1)
            .std()
            .fillna(0.0)
        )

        rolling_mean_price = data["price"].rolling(self.vol_window, min_periods=1).mean()
        rolling_std_price = data["price"].rolling(self.vol_window, min_periods=1).std().replace(
            0, np.nan
        )
        data["price_zscore"] = (
            (data["price"] - rolling_mean_price) / rolling_std_price
        ).fillna(0.0)

        # Optional variance feature if present
        if "variance" in data.columns:
            data["sqrt_variance"] = data["variance"].clip(lower=0.0).pow(0.5)

        data = data.fillna(0.0)
        return data
