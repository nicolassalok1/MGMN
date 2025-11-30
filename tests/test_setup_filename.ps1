Param()
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$setScript = Join-Path $repoRoot "set_me.ps1"
$runScript = Join-Path $repoRoot "run_me.ps1"
$oldScript = Join-Path $repoRoot "setup.ps1"
$readmeSetup = Join-Path $repoRoot "README_SETUP.md"

$failures = @()
function Add-Failure { param([string]$Message) $script:failures += $Message }
function Parse-Errors {
    param([string]$Path)
    $parseErrors = @()
    [System.Management.Automation.Language.Parser]::ParseFile($Path, [ref]$null, [ref]$parseErrors) | Out-Null
    return $parseErrors
}

Write-Host "[INFO] Checking presence and syntax..."
if (-not (Test-Path $setScript)) { Add-Failure "Missing $setScript" }
if (-not (Test-Path $runScript)) { Add-Failure "Missing $runScript" }
if (Test-Path $setScript) {
    $errs = @(Parse-Errors $setScript)
    if ($errs.Count -gt 0) { Add-Failure "set_me.ps1 has parse errors: $($errs | ForEach-Object { $_.Message } | Select-Object -Unique -Join '; ')" }
}
if (Test-Path $runScript) {
    $errs = @(Parse-Errors $runScript)
    if ($errs.Count -gt 0) { Add-Failure "run_me.ps1 has parse errors: $($errs | ForEach-Object { $_.Message } | Select-Object -Unique -Join '; ')" }
}

Write-Host "[INFO] Checking references..."
if (Test-Path $runScript) {
    $runContent = Get-Content -Path $runScript -Raw
    if ($runContent -notmatch "set_me\.ps1") { Add-Failure "run_me.ps1 does not reference set_me.ps1" }
    if ($runContent -match "setup\.ps1") { Add-Failure "run_me.ps1 still references setup.ps1" }
}

if (Test-Path $readmeSetup) {
    $readmeContent = Get-Content -Path $readmeSetup -Raw
    if ($readmeContent -notmatch "set_me\.ps1") { Add-Failure "README_SETUP.md does not mention set_me.ps1" }
    if ($readmeContent -match "setup\.ps1") { Add-Failure "README_SETUP.md still mentions setup.ps1" }
} else {
    Add-Failure "Missing README_SETUP.md at $readmeSetup"
}

Write-Host "[INFO] Checking old script absence..."
if (Test-Path $oldScript) { Add-Failure "Deprecated script still present: $oldScript" }

if ($failures.Count -gt 0) {
    Write-Host "[FAIL] setup rename checks failed:"
    foreach ($f in $failures) { Write-Error $f }
    exit 1
}

Write-Host "[OK] setup rename looks consistent across scripts and docs."
exit 0
