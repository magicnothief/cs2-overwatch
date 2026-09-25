#!/bin/sh
# Install Overwatch review, the CS2 demo reviewer, for the current user (Linux).
#
#   curl -LsSf https://raw.githubusercontent.com/magicnothief/cs2-overwatch/master/install.sh | sh
#
# It gets uv (the tool manager the app installs with) if it is missing, then the
# app, into your home folder: no root, nothing system-wide. The models (2.8 GB)
# download on first start. Uninstall: uv tool uninstall cs2-overwatch, then
# delete ~/.local/share/cs2-overwatch.
set -eu

SOURCE="https://github.com/magicnothief/cs2-overwatch/archive/refs/heads/master.zip"

if ! command -v uv >/dev/null 2>&1; then
    echo "Installing uv, which installs and updates the app..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    PATH="$HOME/.local/bin:$PATH"
fi

uv tool install --python 3.12 --force "cs2-overwatch @ $SOURCE"
uv tool update-shell >/dev/null 2>&1 || true

echo
echo "Installed. Start it with:  overwatch"
echo "(if that command is not found, open a new terminal first)"
