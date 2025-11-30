Param(
    [string]$EnvName = "polyo-gpu"
)

$ErrorActionPreference = "Stop"
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
Set-Location $scriptDir

$tests = @(
    "LSTM-Crypto-Price-Prediction\test_lstm_module.py",
    "tests\test_setup_filename.ps1",
    "tests\test_polyo_pipeline.py",
    "tests\use_case_simple.py",
    "tests\use_case_stress_tests.py",
    "tests\test_input_stress.py",
    "tests\test_telegram_signal_pipeline_unit.py"
)

$conda = Get-Command conda -ErrorAction SilentlyContinue

function Ensure-Pytest {
    if ($conda) {
        Write-Host "Ensuring pytest is installed in '$EnvName'..."
        conda run -n $EnvName python -m pip install pytest -q
    } else {
        Write-Host "Ensuring pytest is installed in current environment..."
        python -m pip install pytest -q
    }
}

function Log {
    param(
        [Parameter(Mandatory=$true)][ValidateSet("INFO","SUCCESS","WARNING","ERROR")]$Level,
        [Parameter(Mandatory=$true)][string]$Message
    )
    $color = switch ($Level) {
        "INFO"    { "Gray" }
        "SUCCESS" { "Green" }
        "WARNING" { "Yellow" }
        "ERROR"   { "Red" }
    }
    Write-Host "[$Level] $Message" -ForegroundColor $color
}

function Build-TestCommand {
    param([string]$FullPath)
    if ($FullPath.ToLower().EndsWith(".ps1")) {
        return @("pwsh", @("-NoProfile","-ExecutionPolicy","Bypass","-File",$FullPath))
    }
    if ($conda) {
        return @("conda", @("run","-n",$EnvName,"python",$FullPath))
    }
    Log -Level "WARNING" -Message "conda not found, using current python for $FullPath"
    return @("python", @($FullPath))
}

$results = @()

function Run-Test {
    param([string]$Path)
    $full = Join-Path $scriptDir $Path
    if (-not (Test-Path $full)) {
        Log -Level "WARNING" -Message "Test not found: $full"
        $script:results += [pscustomobject]@{ Test=$Path; Status="WARNING"; ExitCode=$null }
        return
    }
    $usePytest = $Path -like "*test_telegram_signal_pipeline_unit.py"
    Log -Level "INFO" -Message "Running $Path"
    if ($usePytest) {
        Ensure-Pytest
        if ($conda) {
            conda run -n $EnvName python -m pytest $full
        } else {
            Log -Level "WARNING" -Message "conda not found, using current python for pytest."
            python -m pytest $full
        }
        $exit = $LASTEXITCODE
    } else {
        $cmd = Build-TestCommand -FullPath $full
        & $cmd[0] @($cmd[1])
        $exit = $LASTEXITCODE
    }

    if ($exit -eq 0) {
        Log -Level "SUCCESS" -Message "$Path OK"
        $script:results += [pscustomobject]@{ Test=$Path; Status="SUCCESS"; ExitCode=$exit }
    } else {
        Log -Level "ERROR" -Message "$Path exited with code $exit"
        $script:results += [pscustomobject]@{ Test=$Path; Status="ERROR"; ExitCode=$exit }
    }
}

foreach ($t in $tests) {
    Run-Test -Path $t
}

$succ = ($results | Where-Object { $_.Status -eq "SUCCESS" }).Count
$warn = ($results | Where-Object { $_.Status -eq "WARNING" }).Count
$fail = ($results | Where-Object { $_.Status -eq "ERROR" }).Count
$total = $results.Count

Log -Level "INFO" -Message "Summary: total=$total success=$succ warning=$warn error=$fail"

if ($fail -gt 0) { exit 1 }
exit 0
