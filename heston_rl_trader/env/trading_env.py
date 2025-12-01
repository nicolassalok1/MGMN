import numpy as np
import gymnasium as gym
from gymnasium import spaces

from heston_rl_trader.features.state_builder import StateBuilder


class TradingEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        features_df,
        state_builder: StateBuilder,
        transaction_cost: float = 1e-4,
    ):
        super().__init__()
        self.features_df = features_df.reset_index(drop=True)
        self.state_builder = state_builder.fit(self.features_df)
        self.transaction_cost = transaction_cost
        if self.state_builder.state_dim == 0:
            raise ValueError("State dimension is zero; check feature columns.")

        self.action_space = spaces.Discrete(3)  # 0: short, 1: flat, 2: long
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.state_builder.state_dim,),
            dtype=np.float32,
        )

        self._start_index = max(self.state_builder.window - 1, 0)
        self.reset(seed=None)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = self._start_index
        self.position = 0
        self.nav = 1.0
        self.trades = 0
        obs = self.state_builder.build_state(self.features_df, self.current_step)
        return obs, {}

    def step(self, action: int):
        assert self.action_space.contains(action), "Invalid action"
        target_position = action - 1  # map 0/1/2 -> -1/0/1

        cost = self.transaction_cost * abs(target_position - self.position)
        self.position = target_position
        self.trades += int(cost > 0)

        next_step = self.current_step + 1
        log_return = float(self.features_df.loc[next_step, "log_return"])
        reward = self.position * log_return - cost
        self.nav *= float(np.exp(reward))

        terminated = next_step >= len(self.features_df) - 1
        self.current_step = next_step
        obs = self.state_builder.build_state(self.features_df, self.current_step)
        info = {
            "nav": self.nav,
            "step": self.current_step,
            "position": self.position,
            "trades": self.trades,
        }
        return obs, reward, terminated, False, info

    def render(self):
        print(f"Step {self.current_step}, NAV {self.nav:.4f}, position {self.position}")
