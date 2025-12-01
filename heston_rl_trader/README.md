Raw Market Data
├── Shitcoin Feature Module (pseudo-surface → Heston embedding)
├── BTC Heston Module (IV surface réelle → Heston params)
├── Sentiment Module
└── Generic OHLCV Module
↓
FeatureEngine (fusion)
↓
StateBuilder (normalisation + stacking temporel)
↓
RL Agent (PPO)
↓
TradingEnv (backtest / simulated / live)


---

## 📦 Structure du projet



heston_rl_trader/
├─ models/
│ └─ heston_inverse_model.py
├─ features/
│ ├─ feature_engine.py
│ └─ state_builder.py
├─ data/
│ └─ simulated_data.py
├─ env/
│ └─ trading_env.py
├─ rl/
│ └─ ppo_agent.py
├─ train_ppo.py
└─ requirements.txt


---

## 🚀 Installation



git clone <votre_repo>
cd heston_rl_trader
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt


---

## ▶️ Entraînement PPO

Le script `train_ppo.py` :

- génère un marché simulé BTC + Shitcoin,
- initialise les inverseurs Heston (dummy si pas de poids),
- construit le pipeline complet,
- lance un entraînement PPO full RL.



python train_ppo.py


---

## 🔥 Remplacer les inverseurs Heston

Dans `train_ppo.py` :

```python
btc_model = load_heston_inverse_model(
    nk=5, nt=4, ckpt_path="models/btc_heston.ckpt"
)


Téléchargez/entraînez vos poids et placez-les dans le dossier models/.

📁 Données réelles

Remplacez facilement simulated_data.py par un loader réel
(Crypto/Deribit/FTX/Binance/on-chain).

Les modules sont isolés → zéro friction.

🏗 Roadmap

 Ajouter les contraintes de risque (vol targeting, max leverage).

 Layer de sentiment réel (BERT/distilBERT).

 Calibration Heston réelle sur surface Deribit.

 Passage GPU complet du pipeline (entirely on CUDA).

 Intégration backtest live.

License

MIT License.


---

# 2. `.gitignore` (complet, pro)

```gitignore
# Python
__pycache__/
*.pyc
*.pyo
*.pyd

# Environnements
venv/
.env/
*.env

# Logs
*.log
logs/
wandb/

# Checkpoints / Poids
*.ckpt
*.pt
*.pth
models/*.pt
models/*.pth
models/*.ckpt

# Notebooks
.ipynb_checkpoints/

# Data
data/*.csv
data/*.npz
data/cache/
data/*.pickle
*.npy

# PyTorch / Lightning
lightning_logs/
tensorboard/

# OS
.DS_Store
Thumbs.db