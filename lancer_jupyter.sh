#!/usr/bin/env bash
# Ouvre JupyterLab (les notebooks). Lancement :  bash lancer_jupyter.sh
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
# Moins de fragmentation de la memoire video (utile sur une carte de 4 Go).
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "Demarrage de JupyterLab ..."
echo "Copie l'adresse http://127.0.0.1:8888/lab?token=... affichee ci-dessous"
echo "et ouvre-la dans le navigateur Windows. Puis : dossier notebooks > notebook 01."
echo "Pour arreter : Ctrl+C deux fois."
echo "Astuce 4 Go : ferme les notebooks termines (Kernel > Shut Down) avant d'en ouvrir un autre."
exec jupyter lab --no-browser --ip=127.0.0.1 --port=8888
