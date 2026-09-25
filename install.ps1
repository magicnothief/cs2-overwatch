# Install Overwatch review, the CS2 demo reviewer, for the current user (Windows).
#
#   powershell -ExecutionPolicy ByPass -c "irm https://raw.githubusercontent.com/magicnothief/cs2-overwatch/master/install.ps1 | iex"
#
# It gets uv (the tool manager the app installs with) if it is missing, then the
# app, into your user folder: no administrator rights needed. The models (2.8 GB)
# download on first start. Uninstall: uv tool uninstall cs2-overwatch, then
# delete %LOCALAPPDATA%\cs2-overwatch.
$ErrorActionPreference = "Stop"

$Source = "https://github.com/magicnothief/cs2-overwatch/archive/refs/heads/master.zip"

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Host "Installing uv, which installs and updates the app..."
    powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
    $env:Path = "$env:USERPROFILE\.local\bin;$env:Path"
}

uv tool install --python 3.12 --force "cs2-overwatch @ $Source"
uv tool update-shell | Out-Null

Write-Host ""
Write-Host "Installed. Start it with:  overwatch"
Write-Host "(if that command is not found, open a new terminal first)"
