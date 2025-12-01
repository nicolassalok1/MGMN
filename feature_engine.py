from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import torch
import abc


# =============================================================================
# 1. Abstractions de base
# =============================================================================

class FeatureModule(abc.ABC):
    """
    Interface de base pour un module de features.
    Chaque module prend un "context" (dict de données brutes) et renvoie un dict de features {str: float}.
    """

    @abc.abstractmethod
    def compute_features(self, context: Dict[str, Any]) -> Dict[str, float]:
        raise NotImplementedError


class FeatureEngine:
    """
    Orchestration de modules de features + fusion en vecteur.
    """

    def __init__(self, modules: Dict[str, FeatureModule]):
        """
        modules: dict {nom_module: instance_de_FeatureModule}
        """
        self.modules = modules
        self.feature_order: Optional[List[str]] = None  # ordre stable des features

    def compute_features(
        self,
        context: Dict[str, Any],
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        context: dict global contenant les sous-contextes pour chaque module.
                 ex:
                 {
                    "shitcoin": {...},
                    "btc": {...},
                    "sentiment": {...},
                    "generic": {...},
                 }

        Retourne:
            vector: np.ndarray[float32] de dimension D (features RL-ready, avant normalisation)
            merged: dict complet des features nommés (utile pour debug/log)
        """
        merged: Dict[str, float] = {}

        for name, module in self.modules.items():
            sub_ctx = context.get(name, {})
            feats = module.compute_features(sub_ctx)
            # Préfixe pour éviter collisions: "<module>.<feature>"
            for k, v in feats.items():
                merged[f"{name}.{k}"] = float(v)

        if self.feature_order is None:
            # On fige l'ordre des features une fois pour toutes
            self.feature_order = sorted(merged.keys())

        # Génère le vecteur dans l'ordre figé
        vec = np.array([merged[k] for k in self.feature_order], dtype=np.float32)
        return vec, merged


# =============================================================================
# 2. SHITCOIN MODULE
#    Heston-embedding sur fenêtres glissantes
# =============================================================================

@dataclass
class ShitcoinWindowData:
    prices: np.ndarray          # shape [W]
    volumes: np.ndarray         # shape [W]
    funding: Optional[np.ndarray] = None  # shape [W] ou None
    # tu peux ajouter orderbook snapshot, etc.


class ShitcoinFeatureModule(FeatureModule):
    """
    Module de features pour un shitcoin:
      - stats de fenêtre (vol, skew, volume, funding...)
      - pseudo-surface de moments
      - Heston-embedding via NN inverse
    """

    def __init__(
        self,
        heston_inverse_model: torch.nn.Module,
        device: torch.device,
        maturities: List[int] = [3, 10, 30],
    ):
        self.model = heston_inverse_model
        self.device = device
        self.maturities = maturities
        self.prev_heston: Optional[Dict[str, float]] = None

        self.model.eval()  # on s'assure d'être en mode eval

    # ---------- helpers internes ----------

    @staticmethod
    def _compute_window_stats(window: ShitcoinWindowData) -> Dict[str, float]:
        prices = window.prices
        volumes = window.volumes
        funding = window.funding

        log_ret = np.diff(np.log(prices))
        if len(log_ret) == 0:
            # fenêtre dégénérée -> features neutres
            return {
                "ret_mean": 0.0,
                "realized_vol": 0.0,
                "realized_skew": 0.0,
                "realized_kurt": 0.0,
                "vol_sum": float(volumes.sum()),
                "funding_mean": float(funding.mean() if funding is not None else 0.0),
                "funding_std": float(funding.std() if funding is not None else 0.0),
            }

        ret_mean = log_ret.mean()
        ret_std = log_ret.std()
        realized_vol = float(np.sqrt((log_ret**2).sum()))

        if ret_std > 1e-8:
            realized_skew = float(((log_ret - ret_mean) ** 3).mean() / (ret_std**3))
            realized_kurt = float(((log_ret - ret_mean) ** 4).mean() / (ret_std**4))
        else:
            realized_skew = 0.0
            realized_kurt = 0.0

        vol_sum = float(volumes.sum())
        funding_mean = float(funding.mean() if funding is not None else 0.0)
        funding_std = float(funding.std() if funding is not None else 0.0)

        return {
            "ret_mean": float(ret_mean),
            "realized_vol": realized_vol,
            "realized_skew": realized_skew,
            "realized_kurt": realized_kurt,
            "vol_sum": vol_sum,
            "funding_mean": funding_mean,
            "funding_std": funding_std,
        }

    def _build_pseudo_surface(self, prices: np.ndarray) -> np.ndarray:
        """
        Construit une pseudo-surface de moments sur différentes maturités.
        shape = [len(maturities), 4]
        """
        log_ret = np.diff(np.log(prices))
        surface = []
        for m in self.maturities:
            if len(log_ret) < m:
                r_m = log_ret
            else:
                r_m = log_ret[-m:]
            if len(r_m) == 0:
                mean_m = 0.0
                std_m = 0.0
                skew_m = 0.0
                kurt_m = 0.0
            else:
                mean_m = r_m.mean()
                std_m = r_m.std()
                if std_m > 1e-8:
                    skew_m = ((r_m - mean_m) ** 3).mean() / (std_m**3)
                    kurt_m = ((r_m - mean_m) ** 4).mean() / (std_m**4)
                else:
                    skew_m = 0.0
                    kurt_m = 0.0
            surface.append([mean_m, std_m, skew_m, kurt_m])
        return np.array(surface, dtype=np.float32)

    def _heston_embedding_from_window(self, prices: np.ndarray) -> Dict[str, float]:
        pseudo_surf = self._build_pseudo_surface(prices)  # [M,4]
        # reshape en [B, C, H, W] pour un CNN: B=1, C=1
        x = torch.from_numpy(pseudo_surf).unsqueeze(0).unsqueeze(0)  # [1,1,M,4]
        x = x.to(self.device, dtype=torch.float32)
        with torch.no_grad():
            params_scaled, _ = self.model(x)  # [1,5] si tu as (params, surface_recon)
        params_scaled = params_scaled.squeeze(0).cpu().numpy()
        kappa_s, theta_s, sigma_s, rho_s, v0_s = params_scaled.tolist()
        return {
            "kappa_s": float(kappa_s),
            "theta_s": float(theta_s),
            "sigma_s": float(sigma_s),
            "rho_s": float(rho_s),
            "v0_s": float(v0_s),
        }

    # ---------- interface publique ----------

    def compute_features(self, context: Dict[str, Any]) -> Dict[str, float]:
        """
        context attendu :
        {
            "prices": np.ndarray[W],
            "volumes": np.ndarray[W],
            "funding": Optional[np.ndarray[W]],
        }
        """
        prices = context["prices"]
        volumes = context["volumes"]
        funding = context.get("funding", None)

        window = ShitcoinWindowData(
            prices=np.asarray(prices, dtype=np.float32),
            volumes=np.asarray(volumes, dtype=np.float32),
            funding=np.asarray(funding, dtype=np.float32) if funding is not None else None,
        )

        stats = self._compute_window_stats(window)
        heston = self._heston_embedding_from_window(window.prices)

        feats: Dict[str, float] = {}
        feats.update(stats)
        feats.update(heston)

        # deltas de params Heston
        if self.prev_heston is not None:
            for k, v in heston.items():
                feats[f"d_{k}"] = float(v - self.prev_heston[k])
        else:
            for k in heston.keys():
                feats[f"d_{k}"] = 0.0

        self.prev_heston = heston
        return feats


# =============================================================================
# 3. BTC MODULE
#    Heston sur vraie surface IV BTC
# =============================================================================

@dataclass
class BtcSnapshotData:
    prices: np.ndarray          # historique des prix BTC (pour realized vol)
    idx: int                    # index temps courant
    future_price: float
    funding_rate: float
    open_interest: float
    iv_surface: np.ndarray      # [NK, NT]
    k_grid: np.ndarray          # [NK]
    t_grid: np.ndarray          # [NT]


class BtcHestonFeatureModule(FeatureModule):
    """
    Module de features pour BTC:
      - spot features (ret, realized vol)
      - futures/basis/funding/OI
      - IV features (atm, slope)
      - Heston params sur surface IV
    """

    def __init__(self, heston_inverse_model: torch.nn.Module, device: torch.device):
        self.model = heston_inverse_model
        self.device = device
        self.prev_heston: Optional[Dict[str, float]] = None
        self.model.eval()

    # ---------- helpers ----------

    @staticmethod
    def _compute_spot_features(prices: np.ndarray, idx: int,
                               short_window: int = 30,
                               long_window: int = 240) -> Dict[str, float]:
        p_t = float(prices[idx])
        log_price = float(np.log(p_t))

        idx_short = max(0, idx - short_window)
        idx_long = max(0, idx - long_window)

        ret_short = float(np.log(p_t / prices[idx_short]))
        ret_long = float(np.log(p_t / prices[idx_long]))

        def realized_vol(prices_, start, end):
            if end <= start:
                return 0.0
            log_ret = np.diff(np.log(prices_[start:end+1]))
            return float(np.sqrt((log_ret**2).sum()))

        rv_short = realized_vol(prices, idx_short, idx)
        rv_long = realized_vol(prices, idx_long, idx)

        return {
            "log_price": log_price,
            "ret_short": ret_short,
            "ret_long": ret_long,
            "rv_short": rv_short,
            "rv_long": rv_long,
        }

    @staticmethod
    def _compute_futures_features(spot_price: float,
                                  future_price: float,
                                  funding_rate: float,
                                  open_interest: float) -> Dict[str, float]:
        basis = float(future_price / spot_price - 1.0)
        return {
            "basis": basis,
            "funding_rate": float(funding_rate),
            "open_interest": float(open_interest),
        }

    @staticmethod
    def _compute_iv_features(iv_surface: np.ndarray,
                             k_grid: np.ndarray,
                             t_grid: np.ndarray) -> Dict[str, float]:
        # approximations simples
        atm_idx = int(np.argmin(np.abs(k_grid)))
        short_t_idx = 0
        long_t_idx = -1
        atm_iv_short = float(iv_surface[atm_idx, short_t_idx])
        atm_iv_long = float(iv_surface[atm_idx, long_t_idx])

        put_idx = int(np.argmin(k_grid))    # plus négatif
        call_idx = int(np.argmax(k_grid))   # plus positif
        smile_slope_short = float(iv_surface[put_idx, short_t_idx] -
                                  iv_surface[call_idx, short_t_idx])

        return {
            "atm_iv_short": atm_iv_short,
            "atm_iv_long": atm_iv_long,
            "smile_slope_short": smile_slope_short,
        }

    def _heston_from_iv_surface(self,
                                iv_surface: np.ndarray,
                                k_grid: np.ndarray,
                                t_grid: np.ndarray) -> Dict[str, float]:
        iv = iv_surface.astype(np.float32)
        iv_norm = (iv - iv.mean()) / (iv.std() + 1e-8)
        x = torch.from_numpy(iv_norm).unsqueeze(0).unsqueeze(0)  # [1,1,NK,NT]
        x = x.to(self.device, dtype=torch.float32)
        with torch.no_grad():
            params_scaled, _ = self.model(x)  # [1,5]
        params_scaled = params_scaled.squeeze(0).cpu().numpy()
        kappa_s, theta_s, sigma_s, rho_s, v0_s = params_scaled.tolist()
        return {
            "kappa_s": float(kappa_s),
            "theta_s": float(theta_s),
            "sigma_s": float(sigma_s),
            "rho_s": float(rho_s),
            "v0_s": float(v0_s),
        }

    # ---------- interface publique ----------

    def compute_features(self, context: Dict[str, Any]) -> Dict[str, float]:
        """
        context attendu:
        {
            "prices": np.ndarray,
            "idx": int,
            "future_price": float,
            "funding_rate": float,
            "open_interest": float,
            "iv_surface": np.ndarray[NK,NT],
            "k_grid": np.ndarray[NK],
            "t_grid": np.ndarray[NT],
        }
        """
        data = BtcSnapshotData(
            prices=np.asarray(context["prices"], dtype=np.float32),
            idx=int(context["idx"]),
            future_price=float(context["future_price"]),
            funding_rate=float(context["funding_rate"]),
            open_interest=float(context["open_interest"]),
            iv_surface=np.asarray(context["iv_surface"], dtype=np.float32),
            k_grid=np.asarray(context["k_grid"], dtype=np.float32),
            t_grid=np.asarray(context["t_grid"], dtype=np.float32),
        )

        spot_feats = self._compute_spot_features(data.prices, data.idx)
        fut_feats = self._compute_futures_features(
            float(data.prices[data.idx]),
            data.future_price,
            data.funding_rate,
            data.open_interest,
        )
        iv_feats = self._compute_iv_features(data.iv_surface, data.k_grid, data.t_grid)
        heston = self._heston_from_iv_surface(data.iv_surface, data.k_grid, data.t_grid)

        feats: Dict[str, float] = {}
        feats.update(spot_feats)
        feats.update(fut_feats)
        feats.update(iv_feats)
        feats.update(heston)

        if self.prev_heston is not None:
            for k, v in heston.items():
                feats[f"d_{k}"] = float(v - self.prev_heston[k])
        else:
            for k in heston.keys():
                feats[f"d_{k}"] = 0.0

        self.prev_heston = heston
        return feats


# =============================================================================
# 4. SENTIMENT MODULE (stub propre)
# =============================================================================

class SentimentFeatureModule(FeatureModule):
    """
    Module de sentiment.
    Ici on suppose que le contexte fournit déjà des scores agrégés.
    C'est mieux que d'embarquer tout le NLP dans le même fichier.
    """

    def compute_features(self, context: Dict[str, Any]) -> Dict[str, float]:
        """
        context attendu:
        {
            "sentiment_score": float,
            "msg_rate": float,
            "fear_greed": float,
            ...
        }
        """
        # tu pourrais faire du clipping, normalisation locale, etc. ici.
        feats: Dict[str, float] = {}
        for k, v in context.items():
            feats[k] = float(v)
        return feats


# =============================================================================
# 5. GENERIC MARKET MODULE (OHLCV basique)
# =============================================================================

class GenericMarketFeatureModule(FeatureModule):
    """
    Module de features génériques (OHLCV, etc.).
    À toi de l'enrichir.
    """

    def compute_features(self, context: Dict[str, Any]) -> Dict[str, float]:
        """
        context attendu:
        {
            "close": float,
            "high": float,
            "low": float,
            "volume": float,
            ...
        }
        """
        feats: Dict[str, float] = {}
        for k, v in context.items():
            feats[k] = float(v)
        return feats


# =============================================================================
# 6. Factory : création du FeatureEngine complet
# =============================================================================

def create_default_feature_engine(
    shitcoin_heston_inverse: torch.nn.Module,
    btc_heston_inverse: torch.nn.Module,
    device: torch.device,
) -> FeatureEngine:
    """
    Crée un FeatureEngine avec :
      - module shitcoin (Heston embedding)
      - module BTC (Heston sur surface IV)
      - module sentiment (stub)
      - module generic (OHLCV)
    """
    modules: Dict[str, FeatureModule] = {
        "shitcoin": ShitcoinFeatureModule(shitcoin_heston_inverse, device),
        "btc": BtcHestonFeatureModule(btc_heston_inverse, device),
        "sentiment": SentimentFeatureModule(),
        "generic": GenericMarketFeatureModule(),
    }
    return FeatureEngine(modules)
