#!/usr/bin/env bash
# Verifie Python, bibliotheques et GPU. Lancement :  bash verifier_installation.sh
cd "$(dirname "$(readlink -f "$0")")" || exit 1
if [ ! -f ".venv/bin/activate" ]; then
  echo "Lance d'abord : bash installer.sh"
  exit 1
fi
# shellcheck disable=SC1091
source .venv/bin/activate
export PYTHONUTF8=1
exec python -X utf8 verifier_installation.py
