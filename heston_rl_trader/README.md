# Heston RL Trader

Prototype d'agent PPO pour trader un sous-jacent simulé par un modèle de Heston. Le pipeline génère des prix, extrait des features, construit un état pour l'agent et entraîne un PPO léger en PyTorch.

## Installation
1. Créez un environnement virtuel Python 3.10+ puis activez-le.
2. Installez les dépendances :
   ```bash
   pip install -r requirements.txt
   ```

## Structure
- `models/heston_inverse_model.py` : simulation Heston et estimation rudimentaire de variance réalisée.
- `data/simulated_data.py` : génération de séries synthétiques et split train/test.
- `features/feature_engine.py` : calcul des features (log-return, moyennes mobiles, vol, z-score).
- `features/state_builder.py` : construction de l'état fenêtré pour l'agent.
- `env/trading_env.py` : environnement Gymnasium avec positions short/flat/long et coûts de transaction.
- `rl/ppo_agent.py` : implémentation PPO minimaliste (acteur-critique MLP).
- `train_ppo.py` : script d'entraînement et évaluation simple.

## Lancer un entraînement
Depuis le dossier parent (celui qui contient `heston_rl_trader/`) :
```bash
python -m heston_rl_trader.train_ppo --episodes 5 --steps-per-update 512
```
Arguments utiles :
- `--episodes` : nombre d'épisodes d'entraînement.
- `--steps-per-update` : taille du buffer avant mise à jour des gradients PPO.

## Notes
- Le simulateur Heston utilise une discrétisation d'Euler basique et une calibration très simple sur la variance réalisée. Pour des besoins de recherche ou de production, raffiner la calibration et la génération de trajectoires.
- L'environnement repose sur un mapping discret {-1, 0, 1} (short, flat, long) et récompense par log-return ajusté du coût de transaction.
- Cette base est volontairement légère pour itérer rapidement : ajoutez sauvegarde de modèle, suivi tensorboard et backtests réels selon vos besoins.
