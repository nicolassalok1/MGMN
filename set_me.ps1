Param()
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$env:PYTHONUTF8 = "1"

function Invoke-CmdChecked {
    param(
        [Parameter(Mandatory=$true)][string]$Exe,
        [Parameter()][string[]]$Args = @(),
        [int[]]$AllowedExitCodes = @(0)
    )
    & $Exe @Args
    if ($LASTEXITCODE -notin $AllowedExitCodes) {
        throw "'$Exe $($Args -join ' ')' failed with exit code $LASTEXITCODE"
    }
}

function Require-Command {
    param([string]$Name)
    if (-not (Get-Command $Name -ErrorAction SilentlyContinue)) {
        throw "Required command '$Name' not found. Please install it and re-run."
    }
}

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
Set-Location $scriptDir

Require-Command conda

$envFile = Join-Path $scriptDir "environment.yml"
if (-not (Test-Path $envFile)) {
    throw "environment.yml not found at $envFile"
}

$mlDtypesPin = "ml-dtypes==0.4.0"  # keeps TensorFlow 2.18 compatible while avoiding JAX 0.5.x bump
$jaxVersion = "0.4.28"             # pins JAX to a version that works with ml-dtypes 0.4.x

# Initialize conda for this session
$condaHook = (& conda "shell.powershell" "hook" 2>$null) -join [Environment]::NewLine
if (-not $condaHook) { throw "Could not initialize conda PowerShell hook." }
Invoke-Expression $condaHook

# Create or update environment
$envName = "polyo-gpu"
$existing = conda env list | Select-String "^\s*$envName\s"
$condaPkgs = @("python=3.10","numpy","scipy","pandas","numba","matplotlib","seaborn","scikit-learn","cython","cryptography","streamlit","pip","cmake","ninja","make","pybind11")
if ($existing) {
    Write-Host "Environment '$envName' already exists. Installing/updating core conda packages..."
    $condaArgs = @("install","-n",$envName,"-y") + $condaPkgs
    Invoke-CmdChecked "conda" $condaArgs
} else {
    Write-Host "Creating environment '$envName' with core conda packages..."
    $condaArgs = @("create","-n",$envName,"-y") + $condaPkgs
    Invoke-CmdChecked "conda" $condaArgs
}

conda activate $envName

$extraPaths = @(
    $scriptDir,
    (Join-Path $scriptDir "rough_bergomi"),
    (Join-Path $scriptDir "limit-order-book\python")
) -join ";"
if ($env:PYTHONPATH) {
    $env:PYTHONPATH = "$extraPaths;$env:PYTHONPATH"
} else {
    $env:PYTHONPATH = $extraPaths
}

# Ensure streamlit is available (used by app_gmgn_polyo.py)
Invoke-CmdChecked "conda" @("install","-n",$envName,"-c","conda-forge","streamlit","-y")

# Repair certifi metadata if broken (pip errors about METADATA path)
Invoke-CmdChecked "python" @("-m","pip","install","--force-reinstall","certifi")

# Reinstall pip dependencies listed in environment.yml (pip section)
$pipPkgs = @(
    # Core pins to align TF/JAX/pyarrow
    "numpy==1.26.4",
    "typing-extensions==4.15.0",
    "tensorboard==2.18.0",
    "tensorflow==2.18.0",
    $mlDtypesPin,
    "pyarrow==14.0.2",
    # RL / deps
    "gymnasium==1.1.1",  # matches ray[rllib] constraint
    "pyro-ppl",
    "stable-baselines3",
    "ray[rllib]",
    "requests",
    "tqdm",
    "plotly",
    "openai",
    "telethon"
)
$pipBase = @("-m","pip","install","--upgrade","--no-build-isolation","--progress-bar","off")
foreach ($pkg in $pipPkgs) {
    Invoke-CmdChecked "python" ($pipBase + $pkg) -AllowedExitCodes @(0,120)
}

# GPU detection
$gpuAvailable = $false
if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
    try {
        nvidia-smi > $null 2>&1
        if ($LASTEXITCODE -eq 0) { $gpuAvailable = $true }
    } catch { $gpuAvailable = $false }
}

function Get-PipVersion {
    param([string]$Name)
    $info = (& pip show $Name 2>$null)
    if (-not $info) { return $null }
    $line = $info | Select-String "^Version:"
    if (-not $line) { return $null }
    return ($line.ToString().Split(":")[1]).Trim()
}

function Ensure-Torch {
    param([bool]$UseGpu)
    $torchVer = Get-PipVersion "torch"
    $tvVer = Get-PipVersion "torchvision"
    $taVer = Get-PipVersion "torchaudio"
    $hasAll = $torchVer -and $tvVer -and $taVer
    $torchHasCuda = $torchVer -and ($torchVer -like "*+cu*")
    $targetIndex = $UseGpu ? "https://download.pytorch.org/whl/cu121" : "https://download.pytorch.org/whl/cpu"

    $needsInstall = $true
    if ($hasAll) {
        if ($UseGpu -and $torchHasCuda) { $needsInstall = $false }
        elseif (-not $UseGpu -and -not $torchHasCuda) { $needsInstall = $false }
    }

    if ($needsInstall) {
        Write-Host "Installing PyTorch stack from $targetIndex ..."
        Invoke-CmdChecked "pip" @("install","--upgrade","--index-url",$targetIndex,"torch","torchvision","torchaudio")
    } else {
        Write-Host "PyTorch/vision/audio already match target; skipping reinstall."
    }
}

Ensure-Torch -UseGpu:$gpuAvailable

# Install JAX
$isLinuxOrWSL = -not $IsWindows
if ($gpuAvailable -and $isLinuxOrWSL) {
    Write-Host "Installing JAX $jaxVersion with CUDA 12 support..."
    Invoke-CmdChecked "pip" @("install","--upgrade","jax[cuda12]==$jaxVersion","-f","https://storage.googleapis.com/jax-releases/jax_cuda_releases.html")
} else {
    Write-Host "Installing JAX CPU build ($jaxVersion)..."
    Invoke-CmdChecked "pip" @("install","--upgrade","jax[cpu]==$jaxVersion")
}

# Core Python libs (top-ups)
Invoke-CmdChecked "pip" @(
    "install","--upgrade","--progress-bar","off",
    "numpy==1.26.4","pyarrow==14.0.2","gymnasium==1.1.1","tensorboard==2.18.0",$mlDtypesPin,
    "pyro-ppl","stable-baselines3","ray[rllib]","requests","tqdm","plotly",
    "openai","telethon"
) -AllowedExitCodes @(0,120)
function Install-Editable {
    param(
        [string]$Path,
        [string]$FallbackPip = "",
        [switch]$SkipOnFail
    )
    if (-not (Test-Path $Path)) { throw "Path not found: $Path" }
    $setup = Join-Path $Path "setup.py"
    $pyproject = Join-Path $Path "pyproject.toml"
    if (-not ((Test-Path $setup) -or (Test-Path $pyproject))) {
        Write-Warning "Skipping $Path (no setup.py or pyproject.toml found)."
        return
    }
    Write-Host "python -m pip install -e $Path (no-build-isolation)"
    Push-Location $Path
    try {
        Invoke-CmdChecked "python" @("-m","pip","install","-e",".","--no-build-isolation")
    } catch {
        Write-Warning "Editable install failed for $Path : $($_.Exception.Message)"
        if ($FallbackPip) {
            Write-Warning "Attempting fallback pip install '$FallbackPip' ..."
            try {
                Invoke-CmdChecked "python" @("-m","pip","install",$FallbackPip,"--no-build-isolation")
                return
            } catch {
                Write-Warning "Fallback pip install '$FallbackPip' also failed: $($_.Exception.Message)"
            }
        }
        if (-not $SkipOnFail) { throw }
        Write-Warning "Continuing despite failure for $Path (SkipOnFail)."
    } finally {
        Pop-Location
    }
}

Install-Editable "./rough_bergomi"
Install-Editable "./jumpdiff"
# Prefer prebuilt wheel for hmmlearn to avoid local C++ build issues
Invoke-CmdChecked "python" @("-m","pip","install","--upgrade","hmmlearn","--no-build-isolation")
Install-Editable "./pykalman"
Install-Editable "./RLTrader"
Install-Editable "./TradeMaster"
Install-Editable "./Calibrating-Rough-Volatility-Models-with-Deep-Learning"

# Build limit-order-book C++ engine and Python bindings (MSVC only)
$hasMsvc = Get-Command cl -ErrorAction SilentlyContinue
$lobBuilt = $false
if (-not $hasMsvc) {
    Write-Warning "No MSVC compiler (cl) detected. Skipping limit-order-book build. Ouvre une Developer PowerShell for VS 2022 avec les Build Tools installés puis relance set_me.ps1."
} else {
    Require-Command cmake
    Require-Command ninja
    # Prefer 64-bit MSVC if available (avoid Hostx86/x86 mismatch with 64-bit Python)
    $clCmd = Get-Command cl -ErrorAction SilentlyContinue
    $cmakeCompilerArgs = @()
    # Ensure Windows SDK x64 bin/libs are ahead
    $sdkRoot = "C:\Program Files (x86)\Windows Kits\10"
    if (Test-Path $sdkRoot) {
        $sdkVersions = @(Get-ChildItem -Path (Join-Path $sdkRoot "Lib") -Directory | Sort-Object Name -Descending)
        if ($sdkVersions -and $sdkVersions.Count -gt 0) {
            $sdkVersion = $sdkVersions[0].Name
            $sdkBinX64 = Join-Path $sdkRoot "bin\$sdkVersion\x64"
            if (Test-Path $sdkBinX64) {
                $env:PATH = "$sdkBinX64;$env:PATH"
            }
            $env:WindowsSdkDir = "$sdkRoot\"
            $env:WindowsSDKLibVersion = "$sdkVersion\"
            $cmakeCompilerArgs += "-DCMAKE_SYSTEM_VERSION=$sdkVersion"
            Write-Host "Using Windows SDK $sdkVersion (x64)"
            $sdkUmLib = Join-Path $sdkRoot "Lib\$sdkVersion\um\x64"
            $sdkUcrtLib = Join-Path $sdkRoot "Lib\$sdkVersion\ucrt\x64"
            $libParts = @()
            if (Test-Path $sdkUcrtLib) { $libParts += $sdkUcrtLib }
            if (Test-Path $sdkUmLib) { $libParts += $sdkUmLib }
            # Add MSVC x64 libs
            if ($env:VCToolsInstallDir) {
                $msvcLibX64 = Join-Path $env:VCToolsInstallDir "lib\x64"
                if (Test-Path $msvcLibX64) { $libParts += $msvcLibX64 }
            }
            if ($libParts.Count -gt 0) {
                $env:LIB = ($libParts -join ";") + ";" + $env:LIB
                Write-Host "LIB updated with x64 SDK/MSVC libs: $($libParts -join ';')"
            }
        }
    }
    if ($clCmd) {
        $clPath = $clCmd.Source
        $cl64Path = $null
        if ($clPath -match "Hostx86\\x86") {
            $cl64Candidate = $clPath -replace "Hostx86\\x86","Hostx64\\x64"
            if (Test-Path $cl64Candidate) { $cl64Path = $cl64Candidate }
        }
        if (-not $cl64Path) {
            $cl64Candidate2 = $clPath -replace "Hostx64\\x64","Hostx64\\x64" # no-op fallback
            if (Test-Path $cl64Candidate2) { $cl64Path = $cl64Candidate2 }
        }
        if ($cl64Path) {
            $cmakeCompilerArgs = @("-DCMAKE_C_COMPILER=$cl64Path","-DCMAKE_CXX_COMPILER=$cl64Path")
            Write-Host "Using MSVC compiler: $cl64Path"
        } else {
            Write-Host "Using default MSVC compiler from PATH: $clPath"
        }
    }

    Push-Location (Join-Path $scriptDir "limit-order-book/cpp")
    try {
        New-Item -ItemType Directory -Force -Path "build" | Out-Null
        Push-Location "build"
        try {
            $cmakeArgs = @("-G","Ninja","-DCMAKE_BUILD_TYPE=Release") + $cmakeCompilerArgs + ".."
            Invoke-CmdChecked "cmake" $cmakeArgs
            Invoke-CmdChecked "ninja"

            $lobPython = Join-Path $scriptDir "limit-order-book/python"
            $lobSetup = Join-Path $lobPython "setup.py"
            $lobPyproject = Join-Path $lobPython "pyproject.toml"
            if ((Test-Path $lobSetup) -or (Test-Path $lobPyproject)) {
                Push-Location $lobPython
                try {
                    Invoke-CmdChecked "pip" @("install",".")
                    $lobBuilt = $true
                } finally {
                    Pop-Location
                }
            } else {
                Write-Warning "limit-order-book/python has no setup.py/pyproject.toml; skipping pip install."
            }
        } catch {
            $lobBuilt = $false
            Write-Warning "limit-order-book build failed (CMake/Ninja). Utilise une Developer PowerShell for VS 2022 (MSVC) puis relance set_me.ps1. Détail: $($_.Exception.Message)"
        } finally {
            Pop-Location
        }
    } finally {
        Pop-Location
    }
}

# Sanity checks
Write-Host "Running sanity checks..."
$lobFlag = if ($lobBuilt) { "True" } else { "False" }
$sanity = @"
import sys, os, importlib

# Make sure local workspace paths (set in PYTHONPATH) are on sys.path
for p in os.environ.get("PYTHONPATH", "").split(os.pathsep):
    if p and p not in sys.path:
        sys.path.insert(0, p)

def try_import(name):
    try:
        importlib.import_module(name)
        print(f"[OK] import {name}")
        return True
    except Exception as exc:
        print(f"[FAIL] import {name}: {exc}")
        return False

import torch, jax
modules = [
    "hmmlearn",
    "jumpdiff",
    "pykalman",
    "gymnasium",
    "pandas",
    "numpy",
    "rbergomi",
    "trademaster",
]
results = [try_import(m) for m in modules]
lob_ready = $lobFlag
if lob_ready:
    results.append(try_import("limitorderbook"))
    results.append(try_import("olob"))
else:
    print("[SKIP] limit-order-book imports (build not attempted).")

print("torch.cuda.is_available():", torch.cuda.is_available())
print("torch version:", torch.__version__)
print("jax version:", jax.__version__)
print("python version:", sys.version)
print("All modules import status:", results)
"@
python -c $sanity

Write-Host "`nSetup complete."
