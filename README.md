                           ┌─────────────────────────────────────────┐
                           │             DATA INGESTION              │
                           │─────────────────────────────────────────│
                           │  - Spot prices (all coins)              │
                           │  - Volumes                              │
                           │  - Orderbook snapshots                  │
                           │  - Perps/Futures (funding, basis, OI)   │
                           │  - Options IV surface (BTC only)        │
                           │  - Social feeds (Telegram, Twitter)     │
                           └───────────────────────────┬─────────────┘
                                                       │
                                                       ▼
         ┌──────────────────────────────────────────────────────────────────────────┐
         │                           FEATURE ENGINEERING                            │
         │──────────────────────────────────────────────────────────────────────────│
         │                                                                          │
         │  [1] SHITCOIN REGIME MODULE                                              │
         │      • Sliding windows                                                   │
         │      • Realized vol/skew/kurt                                           │
         │      • Volume/liquidity metrics                                          │
         │      • Funding / OI (if perps)                                           │
         │      • Pseudo-surface build                                              │
         │      • Heston-Embedding NN → (κ,θ,σ,ρ,v0) + Δ-params                     │
         │                                                                          │
         │  [2] BTC HESTON MODULE                                                   │
         │      • BTC IV surface load                                               │
         │      • Surface normalization                                             │
         │      • Inverse-Heston NN → (κ,θ,σ,ρ,v0) + Δ-params                       │
         │      • ATM IV + smile slope + convexity                                  │
         │      • Basis, funding, OI                                                │
         │      • Spot realized vol (multi-horizon)                                 │
         │                                                                          │
         │  [3] SENTIMENT MODULE                                                    │
         │      • Embeddings: BERT/LSTM                                              │
         │      • Sentiment scores                                                  │
         │      • Fear/Greed proxy                                                  │
         │      • Frequency / velocity of messages                                  │
         │                                                                          │
         │  [4] GENERIC MARKET FEATURES                                             │
         │      • OHLCV + return features                                           │
         │      • Market depth, imbalance                                           │
         │      • Trend filters, volatility filters                                 │
         │                                                                          │
         └────────────────────────────┬─────────────────────────────────────────────┘
                                      │
                                      ▼
                     ┌────────────────────────────────────────────────┐
                     │                FEATURE FUSION                  │
                     │────────────────────────────────────────────────│
                     │ Merge dicts/features from:                     │
                     │   • Shitcoin Module                             │
                     │   • BTC Heston Module                           │
                     │   • Sentiment Module                            │
                     │   • Generic Features                            │
                     │ Outputs a single unified feature vector         │
                     └──────────────────────────┬──────────────────────┘
                                                │
                                                ▼
               ┌────────────────────────────────────────────────────────────────┐
               │                           STATE BUILDER                        │
               │────────────────────────────────────────────────────────────────│
               │  • Normalisation (running stats/scaler by feature)             │
               │  • Clipping (anti-outliers)                                    │
               │  • Time-stacking (N past steps → tensor [N, D])                │
               │  • Padding / masking for irregularities                        │
               │  • Final RL-ready state tensor                                 │
               └─────────────────────────────┬──────────────────────────────────┘
                                             │
                                             ▼
                  ┌───────────────────────────────────────────────────────┐
                  │                        RL CORE                        │
                  │───────────────────────────────────────────────────────│
                  │  POLICY NETWORK (PPO / SAC / TD3 / DQN / custom)      │
                  │                                                       │
                  │  Input : STATE[t]                                     │
                  │  Output : ACTION[t] (weights, long/short, leverage…)  │
                  │                                                       │
                  │  CRITIC NETWORK (value estimation)                    │
                  │                                                       │
                  └──────────────────────────────┬────────────────────────┘
                                                 │
                                                 ▼
                      ┌────────────────────────────────────────────┐
                      │        EXECUTION / LIVE TRADING ENGINE     │
                      │────────────────────────────────────────────│
                      │  • Position management                     │
                      │  • Risk constraints (per-coin, global)     │
                      │  • Slippage model                          │
                      │  • Leverage limits                         │
                      │  • Orders → Exchange API                   │
                      └──────────────────────────┬─────────────────┘
                                                 │
                                                 ▼
                          ┌────────────────────────────────────┐
                          │           LOGGING / METRICS        │
                          │────────────────────────────────────│
                          │  - PnL, Sharpe, DD                 │
                          │  - Regime transitions              │
                          │  - Feature drift                   │
                          │  - Latent Heston drift             │
                          │  - Policy diagnostics              │
                          └────────────────────────────────────┘
