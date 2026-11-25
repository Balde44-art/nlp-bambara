#!/usr/bin/env bash
# Ouvre l'interface de traduction (Gradio). Lancement :  bash lancer_interface.sh
cd "$(dirname "$(readlink -f "$0")")" || exit 1
if [ ! -f ".venv/bin/activate" ]; then
  echo "Lance d'abord : bash installer.sh"
  exit 1
fi
# shellcheck disable=SC1091
source .venv/bin/activate
export PYTHONUTF8=1
export HF_HUB_DISABLE_SYMLINKS_WARNING=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "Chargement des modeles, une minute environ ..."
echo "Quand l'adresse http://127.0.0.1:7860 apparait, ouvre-la dans le navigateur Windows."
echo "(Ferme JupyterLab avant : deux programmes ne tiennent pas ensemble dans 4 Go.)"
exec python app/demo_gradio.py
