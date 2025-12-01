
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Optional


# =============================================================================
# 1. RunningStats (online mean / variance)
# =============================================================================

class RunningStats:
    """
    Maintient mean/var en ligne pour les features.
    Implémentation classique (Welford). Stable numériquement.
    """

    def __init__(self, dim: int):
        self.dim = dim
        self.count = 0
        self.mean = np.zeros(dim, dtype=np.float64)
        self.M2 = np.zeros(dim, dtype=np.float64)  # sum of squares of diff

    def update(self, x: np.ndarray):
        """
        x: vecteur [dim]
        """
        x = x.astype(np.float64)
        self.count += 1
        delta = x - self.mean
        self.mean += delta / self.count
        delta2 = x - self.mean
        self.M2 += delta * delta2

    @property
    def var(self):
        if self.count < 2:
            return np.ones(self.dim, dtype=np.float64)
        return self.M2 / (self.count - 1)

    @property
    def std(self):
        return np.sqrt(self.var) + 1e-8   # évite les divisions nulles


# =============================================================================
# 2. Normalizer
# =============================================================================

class Normalizer:
    """
    Normalisation en ligne (z-score) + clipping (optionnel).
    """

    def __init__(self, dim: int, clip_value: float = 5.0):
        self.stats = RunningStats(dim)
        self.clip_value = clip_value

    def normalize(self, x: np.ndarray, update_stats: bool = True) -> np.ndarray:
        """
        x: vecteur de features (non normalisé)
        update_stats: True pendant l'entraînement, False en déploiement live
        """
        if update_stats:
            self.stats.update(x)
        normed = (x - self.stats.mean) / self.stats.std
        np.clip(normed, -self.clip_value, self.clip_value, out=normed)
        return normed.astype(np.float32)


# =============================================================================
# 3. StateBuffer (stack temporel)
# =============================================================================

class StateBuffer:
    """
    Buffer circulaire pour maintenir N historiques des observations normalisées.
    Sorties: tensor shape = [window, dim].
    """

    def __init__(self, window: int, dim: int):
        self.window = window
        self.dim = dim
        self.buffer = np.zeros((window, dim), dtype=np.float32)
        self.ptr = 0
        self.initialized = False

    def reset(self):
        self.buffer.fill(0.0)
        self.ptr = 0
        self.initialized = False

    def push(self, x: np.ndarray):
        """
        x shape = [dim]
        """
        self.buffer[self.ptr] = x
        self.ptr = (self.ptr + 1) % self.window

        if not self.initialized and self.ptr == 0:
            self.initialized = True

    def get_state(self) -> np.ndarray:
        """
        Retourne un tableau temporel ordonné du plus ancien → plus récent :
        shape: [window, dim]
        """
        if not self.initialized:
            # tant que le buffer n'est pas plein, on return ce qu'on a
            # en restant stable
            return self.buffer.copy()
        # recompose le flux temporel circulaire
        indices = np.arange(self.ptr, self.ptr + self.window) % self.window
        return self.buffer[indices].copy()


# =============================================================================
# 4. StateBuilder (ensemble)
# =============================================================================

class StateBuilder:
    """
    Combine normalizer + state buffer pour produire l'état final RL.
    """

    def __init__(
        self,
        dim: int,
        window: int = 16,
        clip_value: float = 5.0,
        training: bool = True,
    ):
        """
        dim: dimension du vecteur de features fusionné
        window: taille de fenêtre temporelle
        training: True si on update les stats (entraînement RL),
                  False en mix-déploiement si tu veux figer les stats
        """
        self.dim = dim
        self.window = window
        self.training = training

        self.normalizer = Normalizer(dim=dim, clip_value=clip_value)
        self.buffer = StateBuffer(window=window, dim=dim)

    def reset(self):
        """
        Reset complet entre épisodes.
        """
        self.buffer.reset()

    def build_state(self, feature_vec: np.ndarray) -> np.ndarray:
        """
        Prend un vecteur de features (non normalisé),
        le normalise, l'empile dans le buffer,
        renvoie le state final shape [window, dim].
        """
        feature_vec = feature_vec.astype(np.float32)

        # normalisation (z-score)
        normed = self.normalizer.normalize(feature_vec, update_stats=self.training)

        # push dans le buffer circulaire
        self.buffer.push(normed)

        # construit le tensor final
        state = self.buffer.get_state()  # shape [window, dim]
        return state
