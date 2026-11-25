#!/usr/bin/env bash
# Tests du detecteur de langue. Lancement :  bash lancer_tests.sh
cd "$(dirname "$(readlink -f "$0")")" || exit 1
if [ ! -f ".venv/bin/activate" ]; then
  echo "Lance d'abord : bash installer.sh"
  exit 1
fi
# shellcheck disable=SC1091
source .venv/bin/activate
exec python -m pytest tests
