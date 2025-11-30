# POLYO Setup Guide (Windows)

Ce document résume l’installation complète pour un clone frais du repo, y compris CUDA, Build Tools, conda, et la résolution des pièges habituels. Objectif : pouvoir lancer `set_me.ps1` puis `run_me.ps1` sans surprises.

## Prérequis système
- Windows 10/11 64‑bit.
- Droits d’installation (admin recommandé).
- Espace disque suffisant (~15‑20 Go).

## Étape 1 — GPU / CUDA (optionnel mais recommandé)
1. Installe les pilotes NVIDIA récents (GeForce/Quadro selon ton GPU).
2. Installe le **CUDA Toolkit** (version compatible avec cu121 pour torch 2.5.1+cu121).
3. Vérifie avec `nvidia-smi` dans un terminal : si la commande répond, le GPU est visible. Sinon, réinstalle les drivers/CUDA.

> Sans GPU, tout fonctionne en CPU, mais plus lent. Le script avertit si `nvidia-smi` est absent.

## Étape 2 — Build Tools (compilateur C++)
Certaines dépendances (limit-order-book) nécessitent un compilateur C++.
1. Installe **Visual Studio Build Tools** ou **Visual Studio Community**.
2. Sélectionne le workload **“Desktop development with C++”** (inclut MSVC, CMake, Ninja).
3. Redémarre PowerShell après installation pour que `cl.exe` soit dans le PATH (ou ouvre un “Developer PowerShell”).
4. Vérifie avec `cl` dans le terminal ; si non trouvé, relance la session ou exécute `VsDevCmd.bat` de VS.

> Si tu n’as pas besoin de `limit-order-book`, tu peux ignorer l’absence de compilateur (warning seulement).

### Builder limit-order-book avec MSVC
- Ouvre une **Developer PowerShell for VS 2022** (menu Démarrer) afin de charger `cl`, CMake et Ninja.
- Vérifie `cl` dans le terminal.
- Dans `D:\PythonDProjects\POLYO`, relance `pwsh -NoProfile -ExecutionPolicy Bypass -File .\set_me.ps1` pour builder `limit-order-book`.
- Pas besoin de rendre ça permanent : relance simplement une Developer PowerShell quand tu veux reconstruire `limit-order-book`. Si tu restes en PowerShell standard, appelle `VsDevCmd.bat` avant le setup.

En résumé pour (re)builder `limit-order-book` :
```pwsh
# Dans une "Developer PowerShell for VS 2022" (menu Démarrer)
cl   # vérifier que MSVC répond
cd D:\PythonDProjects\POLYO
Remove-Item limit-order-book\cpp\build -Recurse -Force -ErrorAction SilentlyContinue  # optionnel
pwsh -NoProfile -ExecutionPolicy Bypass -File .\set_me.ps1
```

Si tu utilises le terminal intégré VS Code, crée/choisis un profil “Developer PowerShell” pour que `cl` soit chargé avant de lancer `set_me.ps1`.

### VS Code : choisir le profil “Developer PowerShell”
1. `Ctrl+Shift+P` → “Terminal: Select Default Profile”.
2. Choisir “Developer PowerShell for VS 2022” (ou “Developer Command Prompt for VS 2022”). Si tu ne le vois pas dans la liste, il faudra l’ajouter manuellement (voir ci-dessous).
3. Ouvrir un nouveau terminal (`Ctrl+Shift+``) : `cl` sera dans le PATH.
4. Si le profil n’apparaît pas, redémarrer VS Code ou la session. En dernier recours, ajouter un profil manuel pointant vers :
   `C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\Launch-VsDevShell.ps1 -Arch amd64 -HostArch amd64`

Ajouter le profil manuellement (settings JSON VS Code) si absent :
```json
"terminal.integrated.profiles.windows": {
  "Developer PowerShell": {
    "path": "C:\\Program Files (x86)\\Microsoft Visual Studio\\2022\\BuildTools\\Common7\\Tools\\Launch-VsDevShell.ps1",
    "args": ["-Arch", "amd64", "-HostArch", "amd64"],
    "icon": "terminal-powershell"
  }
}
```
Puis sélectionne ce profil comme défaut et ouvre un nouveau terminal.

## PowerShell + C++ dans VS Code (profil défaut)
Objectif : ouvrir PowerShell dans le terminal intégré VS Code avec un compilateur C++ prêt à l'emploi.

- Vérifier PowerShell 7 : `pwsh -v`. Si non trouvé, installer PowerShell 7 (MSI depuis https://aka.ms/powershell-release) puis rouvrir VS Code.
- Définir PowerShell comme profil terminal du workspace : `Ctrl+Shift+P` -> "Preferences: Open Workspace Settings (JSON)" et ajoute/modifie :
  ```json
  {
    "folders": [{ "path": "." }],
    "settings": {
      "terminal.integrated.defaultProfile.windows": "PowerShell"
    }
  }
  ```
  Remplace par `"Windows PowerShell"` si tu préfères la v5 intégrée, ou par un profil "Developer PowerShell" si tu utilises MSVC.
- Compiler en C++ depuis PowerShell :
  - MSVC (Build Tools) : ouvre une Developer PowerShell for VS 2022 ou exécute `VcVars64.bat`/`Launch-VsDevShell.ps1`, puis `cl /EHsc /std:c++17 main.cpp`.
 - Contrôle rapide : `cl` doit afficher la version (MSVC). Si la commande est introuvable, relance une Developer PowerShell ou corrige le PATH.

## Étape 3 - Miniconda/Conda
1. Installe **Miniconda** ou **Anaconda** (choisir "Add conda to PATH" ou initialise le shell via `conda init powershell`).
2. Ouvre un PowerShell sans profil, teste `conda --version`.

## Étape 4 — Cloner le repo
```pwsh
git clone <repo> POLYO
cd POLYO
```

## Étape 5 — Provisioning automatique
Lancer le setup complet (installe env `polyo-gpu`, PyTorch CUDA, JAX, streamlit, dépendances locales, etc.) :
```pwsh
pwsh -NoProfile -ExecutionPolicy Bypass -File .\set_me.ps1
```
Ce que fait `set_me.ps1` :
- Crée/actualise l’env conda `polyo-gpu` avec les packages de base + pybind11.
- Installe streamlit via conda-forge.
- Répare certifi si cassé.
- Réinstalle/upgrade les paquets pip clés (gymnasium, tensorboard, pyro-ppl, stable-baselines3, ray[rllib], requests, tqdm, plotly).
- Installe en editable : jumpdiff, pykalman, TradeMaster (et saute ceux sans setup) et installe `hmmlearn` via wheel pour éviter les builds C++ sur Windows.
- Tente de builder limit-order-book (si `cl` MSVC est disponible) en forçant le toolchain x64 (Hostx64/x64), en pointant sur le Windows SDK x64 le plus récent, en préfixant `LIB` avec les libs x64 SDK/MSVC, et en fixant les destinations d’install CMake pour le module pybind (`CMAKE_INSTALL_LIBDIR=lib` et `CMAKE_LIBRARY_OUTPUT_DIRECTORY=lib`). Sinon, warning seulement.
- Fait des sanity checks d’import (hmmlearn, jumpdiff, pykalman, gymnasium, pandas, numpy, rbergomi, trademaster) et affiche torch/jax versions + torch.cuda.is_available().

## Étape 6 — Lancer l’application Streamlit
```pwsh
pwsh -ExecutionPolicy Bypass -File .\run_me.ps1
```
Options :
- `-ApiKey "<GMGN_API_KEY>"` (facultatif). Sans clef → mode TEST (dummy data).
- `-EnvName "polyo-gpu"` (par défaut). Le script crée l’env via `set_me.ps1` si absent, vérifie streamlit, puis lance `python -m streamlit run app_gmgn_polyo.py` via conda run.

## Pièges fréquents & solutions
- **`conda` introuvable** : réouvre un terminal après `conda init powershell`, ou lance depuis “Anaconda Prompt (PowerShell)”. Assure-toi que Miniconda est sur le PATH.
- **`streamlit` introuvable** : `run_me.ps1` l’installera si besoin. Sinon : `conda install -n polyo-gpu -c conda-forge streamlit`.
- **Certifi METADATA manquante** (erreur pip) : `python -m pip install --force-reinstall certifi` (déjà fait dans `set_me.ps1`).
- **Compilateur C++ manquant** : installe Build Tools + workload C++; ou ignore si tu n’as pas besoin de limit-order-book.
- **CUDA non détectée** : vérifier `nvidia-smi`, drivers, version CUDA. Sinon, l’app tourne en CPU.
- **Ray/gym versions** : le setup réinstalle les versions compatibles ; en cas de conflit, nettoyer l’env (`conda env remove -n polyo-gpu`) puis relancer `set_me.ps1`.
- **Build limit-order-book** : lance `set_me.ps1` depuis une **Developer PowerShell for VS 2022** (profil amd64). Le script force l’usage de `Hostx64\\x64\\cl.exe`, du Windows SDK x64 détecté, préfixe `LIB` avec les libs x64 SDK/MSVC, et fixe les destinations d’install CMake (`CMAKE_INSTALL_LIBDIR/lib`, `CMAKE_LIBRARY_OUTPUT_DIRECTORY/lib`) pour éviter l’erreur “install TARGETS given no LIBRARY DESTINATION”. Il continue même si la build est sautée.
- **hmmlearn** : installé via wheel (`pip install hmmlearn`) pour éviter les erreurs de linkage MSVC. Si tu veux builder en editable, ouvre une Developer PowerShell x64 et lance manuellement `pip install -e . --no-build-isolation` dans `hmmlearn/`.
- **Windows 10 SDK** : installe le SDK Windows 10 via Visual Studio Build Tools (onglet “Composants individuels” → “Kit de développement logiciel (SDK) Windows 10”). Sans lui, la build MSVC x64 du limit-order-book peut échouer faute de libs/headers.

## Commandes utiles
- Supprimer l’env et repartir propre :
  ```pwsh
  conda env remove -n polyo-gpu
  pwsh -NoProfile -ExecutionPolicy Bypass -File .\set_me.ps1
  ```
- Vérifier GPU : `nvidia-smi`
- Lancer l’app directement si l’env est actif : `python -m streamlit run app_gmgn_polyo.py`
