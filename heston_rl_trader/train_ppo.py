# train_ppo.py
from __future__ import annotations
import numpy as np

import torch

from models.heston_inverse_model import load_heston_inverse_model
from features.feature_engine import create_default_feature_engine
from features.state_builder import StateBuilder
from data.simulated_data import simulate_market, SimMarketConfig
from env.trading_env import TradingEnv, TradingEnvConfig
from rl.ppo_agent import PPOAgent, PPOConfig, compute_gae


def train_ppo(
    total_steps: int = 50_000,
    rollout_len: int = 2048,
    update_epochs: int = 10,
):

    # 1) Market simulé
    market = simulate_market(SimMarketConfig(n_steps=total_steps + 5_000))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2) Modèles Heston inverses (dummy pour l’instant)
    # Pour BTC, NK=5, NT=4 comme dans simulate_market
    btc_model = load_heston_inverse_model(nk=5, nt=4, ckpt_path=None, device=device)
    # Pour shitcoin, on suppose pseudo-surface [M,4] avec M=3 (maturities)
    # Donc nt=4, nk=3 pour garder la même arch CNN 2D (1, NK=M, NT=4)
    shit_model = load_heston_inverse_model(nk=3, nt=4, ckpt_path=None, device=device)

    # 3) FeatureEngine + StateBuilder
    from features.feature_engine import create_default_feature_engine

    fe = create_default_feature_engine(
        shitcoin_heston_inverse=shit_model,
        btc_heston_inverse=btc_model,
        device=device,
    )

    # on récupère la dimension en faisant un appel factice
    dummy_ctx = {
        "shitcoin": {
            "prices": market.shit_prices[:60],
            "volumes": market.shit_volumes[:60],
            "funding": market.shit_funding[:60],
        },
        "btc": {
            "prices": market.btc_prices[:100],
            "idx": 99,
            "future_price": market.btc_fut_prices[99],
            "funding_rate": market.btc_funding[99],
            "open_interest": market.btc_oi[99],
            "iv_surface": market.btc_iv_surface[99],
            "k_grid": market.k_grid,
            "t_grid": market.t_grid,
        },
        "sentiment": {
            "sentiment_score": 0.0,
            "msg_rate": 0.0,
            "fear_greed": 0.0,
        },
        "generic": {
            "close": float(market.btc_prices[99]),
            "high": float(market.btc_prices[99]),
            "low": float(market.btc_prices[99]),
            "volume": 1.0,
        },
    }
    dummy_vec, _ = fe.compute_features(dummy_ctx)
    dim = dummy_vec.shape[0]

    sb = StateBuilder(dim=dim, window=16, clip_value=5.0, training=True)

    # 4) Environnement
    env_cfg = TradingEnvConfig(
        window_shitcoin=60,
        step_start=100,
        max_steps=total_steps,
        initial_capital=1_000.0,
        transaction_cost=0.0005,
    )
    env = TradingEnv(market=market, feature_engine=fe, state_builder=sb, config=env_cfg)

    # 5) Agent PPO
    obs, info = env.reset()
    obs_dim = obs.shape[0]
    ppo_cfg = PPOConfig()
    agent = PPOAgent(obs_dim=obs_dim, cfg=ppo_cfg)

    ep_return = 0.0
    ep_len = 0
    global_step = 0

    while global_step < total_steps:
        # rollout
        obs_buf = []
        act_buf = []
        logp_buf = []
        rew_buf = []
        val_buf = []
        done_buf = []

        for _ in range(rollout_len):
            obs_t = obs.copy()
            obs_buf.append(obs_t)

            # valeur courante
            with torch.no_grad():
                v_t = agent.v_net(
                    torch.as_tensor(obs_t, dtype=torch.float32, device=agent.device).unsqueeze(0)
                ).cpu().numpy()[0, 0]
            val_buf.append(v_t)

            act, logp, _ = agent.act(obs_t)
            next_obs, reward, terminated, truncated, info = env.step(act)

            act_buf.append(act)
            logp_buf.append(logp)
            rew_buf.append(reward)
            done = float(terminated or truncated)
            done_buf.append(done)

            ep_return += reward
            ep_len += 1
            global_step += 1

            obs = next_obs

            if terminated or truncated:
                print(f"Episode done: return={ep_return:.4f} length={ep_len}")
                obs, info = env.reset()
                ep_return = 0.0
                ep_len = 0

            if global_step >= total_steps:
                break

        # valeur bootstrap
        with torch.no_grad():
            v_last = agent.v_net(
                torch.as_tensor(obs, dtype=torch.float32, device=agent.device).unsqueeze(0)
            ).cpu().numpy()[0, 0]
        val_buf.append(v_last)

        obs_arr = np.array(obs_buf, dtype=np.float32)
        act_arr = np.array(act_buf, dtype=np.float32)
        logp_arr = np.array(logp_buf, dtype=np.float32)
        rew_arr = np.array(rew_buf, dtype=np.float32)
        val_arr = np.array(val_buf, dtype=np.float32)
        done_arr = np.array(done_buf, dtype=np.float32)

        adv, ret = compute_gae(
            rews=rew_arr,
            vals=val_arr,
            dones=done_arr,
            gamma=ppo_cfg.gamma,
            lam=ppo_cfg.lam,
        )

        buf = {
            "obs": obs_arr,
            "act": act_arr,
            "logp": logp_arr,
            "adv": adv,
            "ret": ret,
        }

        agent.update(buf)

    print("Training finished.")


if __name__ == "__main__":
    train_ppo()
