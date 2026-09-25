# Install Overwatch review, the CS2 demo reviewer, for the current user (Windows).
#
#   powershell -ExecutionPolicy ByPass -c "irm https://raw.githubusercontent.com/magicnothief/cs2-overwatch/master/install.ps1 | iex"
#
# It gets uv (the tool manager the app installs with) if it is missing, then the
# latest release of the app, into your user folder: no administrator rights
# needed. The models (2.8 GB) download on first start. Running it again updates
# the app. Uninstall: uv tool uninstall cs2-overwatch, then delete
# %LOCALAPPDATA%\cs2-overwatch.
$ErrorActionPreference = "Stop"

$Repo = "magicnothief/cs2-overwatch"
# the latest release's package; the newest code when nothing is released yet
$Source = "cs2-overwatch @ https://github.com/$Repo/archive/refs/heads/master.zip"
try {
    $Release = Invoke-RestMethod "https://api.github.com/repos/$Repo/releases/latest"
    $Wheel = $Release.assets | Where-Object { $_.name -like "*.whl" } | Select-Object -First 1
    if ($Wheel) { $Source = $Wheel.browser_download_url }
} catch { }

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Host "Installing uv, which installs and updates the app..."
    powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
    $env:Path = "$env:USERPROFILE\.local\bin;$env:Path"
}

uv tool install --python 3.12 --force "$Source"
uv tool update-shell | Out-Null

Write-Host ""
Write-Host "Installed. Start it with:  overwatch"
Write-Host "(if that command is not found, open a new terminal first)"
