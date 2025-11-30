param(
    [string]$ApiKey = "",
    [string]$OpenAIKey = "",
    [string]$EnvName = "polyo-gpu"
)

$setupScriptName = "set_me.ps1"

# Renseigne ton API key ici ou passe-le en argument -ApiKey.
# Si aucune cle n'est fournie, l'app utilisera automatiquement les donnees dummy_gmgn (mode test).

if ($ApiKey -ne "") { $env:GMGN_API_KEY = $ApiKey }
if ($OpenAIKey -ne "") { $env:OPENAI_API_KEY = $OpenAIKey }

$gmgnLen = 0
if ($env:GMGN_API_KEY) { $gmgnLen = $env:GMGN_API_KEY.Length }
$openaiLen = 0
if ($env:OPENAI_API_KEY) { $openaiLen = $env:OPENAI_API_KEY.Length }

Write-Host "GMGN_API_KEY longueur: $gmgnLen"
if ($gmgnLen -eq 0) {
    Write-Warning "Aucune cle GMGN detectee, l'app demarrera en mode TEST (dummy data)."
} else {
    Write-Host "Cle GMGN detectee, tentative de mode LIVE."
}
Write-Host "OPENAI_API_KEY longueur: $openaiLen"
if ($openaiLen -eq 0) {
    Write-Warning "Pas de cle OpenAI; le module Telegram/Sentiment utilisera le mode heuristique."
}

$gpuAvailable = $false
if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
    try {
        nvidia-smi > $null 2>&1
        if ($LASTEXITCODE -eq 0) { $gpuAvailable = $true }
    } catch { $gpuAvailable = $false }
}
if (-not $gpuAvailable) {
    Write-Warning "GPU/CUDA non detecte (nvidia-smi absent). L'app fonctionnera en mode CPU. Installe les pilotes NVIDIA + CUDA Toolkit si tu souhaites l'acceleration GPU."
} else {
    Write-Host "GPU/CUDA detecte via nvidia-smi."
}

$condaCmd = Get-Command conda -ErrorAction SilentlyContinue
if (-not $condaCmd) {
    Write-Warning "conda introuvable; active l'env '$EnvName' manuellement et installe streamlit si besoin."
    Write-Host "Tentative d'executer streamlit directement..."
    streamlit run app_gmgn_polyo.py
    exit 0
}

$envExists = conda env list | Select-String "^\s*$EnvName\s"
if (-not $envExists) {
    Write-Warning "L'environnement '$EnvName' est introuvable. Lancement de set_me.ps1 pour le creer..."
    $setupPath = Join-Path $PSScriptRoot "set_me.ps1"
    & $setupPath
}

$pyCheck = @'
import importlib, sys
try:
    importlib.import_module("streamlit")
    print("streamlit ok")
except Exception as exc:
    print(f"missing_streamlit::{exc}")
    sys.exit(1)
'@

Write-Host "Verification de streamlit dans l'env '$EnvName'..."
$pyCheck | conda run -n $EnvName python -
if ($LASTEXITCODE -ne 0) {
    Write-Host "Installation de streamlit dans '$EnvName'..."
    conda install -n $EnvName -c conda-forge streamlit -y
}

Write-Host "Execution via conda run dans l'env '$EnvName' avec python -m streamlit..."
conda run -n $EnvName python -m streamlit run app_gmgn_polyo.py
