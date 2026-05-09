# Xybrid CLI installer for Windows
# Usage: irm https://raw.githubusercontent.com/xybrid-ai/xybrid/master/install.ps1 | iex
$ErrorActionPreference = "Stop"

$Repo = "xybrid-ai/xybrid"
$BinaryName = "xybrid"

# --- Detect architecture ---

$Arch = [System.Runtime.InteropServices.RuntimeInformation]::OSArchitecture
switch ($Arch) {
    "X64"   { $Platform = "windows-x86_64" }
    default { Write-Error "Unsupported architecture: $Arch. Install from source: cargo install --git https://github.com/$Repo xybrid-cli"; exit 1 }
}

# --- Get latest version ---

Write-Host "==> Xybrid CLI installer" -ForegroundColor Blue
Write-Host ""

# Can't use /releases/latest — it may return cargokit precompiled_* releases.
# Instead, find the first release whose tag starts with "v".
$Releases = Invoke-RestMethod -Uri "https://api.github.com/repos/$Repo/releases" -Headers @{ "User-Agent" = "xybrid-installer" }
$Version = ($Releases | Where-Object { $_.tag_name -match "^v" } | Select-Object -First 1).tag_name

if (-not $Version) {
    Write-Error "Could not determine latest version. Check https://github.com/$Repo/releases"
    exit 1
}

# --- Download ---

$Artifact = "$BinaryName-$Version-$Platform.exe"
$Url = "https://github.com/$Repo/releases/download/$Version/$Artifact"

$InstallDir = "$env:USERPROFILE\.xybrid\bin"
if (-not (Test-Path $InstallDir)) {
    New-Item -ItemType Directory -Path $InstallDir -Force | Out-Null
}

$Dest = Join-Path $InstallDir "$BinaryName.exe"

Write-Host "==> Downloading xybrid $Version for $Platform..." -ForegroundColor Blue

try {
    Invoke-WebRequest -Uri $Url -OutFile $Dest -UseBasicParsing
} catch {
    Write-Error "Download failed. Binary may not exist for this release.`nURL: $Url`nTry: cargo install --git https://github.com/$Repo xybrid-cli"
    exit 1
}

# --- Add to PATH ---

$UserPath = [Environment]::GetEnvironmentVariable("Path", "User")
if ($UserPath -notlike "*$InstallDir*") {
    [Environment]::SetEnvironmentVariable("Path", "$InstallDir;$UserPath", "User")
    Write-Host ""
    Write-Host "==> Added $InstallDir to your PATH." -ForegroundColor Blue
    Write-Host "    Restart your terminal for PATH changes to take effect." -ForegroundColor Yellow
}

Write-Host ""
Write-Host "==> Installed xybrid $Version to $Dest" -ForegroundColor Blue
Write-Host ""
Write-Host "  Get started:"
Write-Host "    xybrid --help"
Write-Host "    xybrid models list"
Write-Host '    xybrid run --model kokoro-82m --input-text "Hello world" -o output.wav'
Write-Host ""
