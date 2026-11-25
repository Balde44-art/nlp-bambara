#!/usr/bin/env bash
# Installation du projet nlp-bambara dans WSL (une seule fois).
# Lancement :  bash installer.sh
set -e
cd "$(dirname "$(readlink -f "$0")")"

echo "============================================================"
echo "  Installation du projet nlp-bambara (WSL)"
echo "============================================================"

# Un projet sur /mnt/c est tres lent et casse parfois les liens symboliques.
case "$PWD" in
  /mnt/*)
    echo
    echo "ATTENTION : le projet est sur le disque Windows ($PWD)."
    echo "C'est tres lent depuis WSL. Copie-le plutot dans ton dossier Linux :"
    echo "    cp -r \"$PWD\" ~/nlp-bambara-wsl && cd ~/nlp-bambara-wsl"
    read -r -p "Continuer quand meme ? (o/N) " rep
    [ "$rep" = "o" ] || [ "$rep" = "O" ] || exit 1
    ;;
esac

# 1. Python

# Python 3.12 + environnement virtuel
PYTHON_BIN="python3.12"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "ERREUR : Python 3.12 est introuvable."
  echo "Installe Python 3.12 avant de continuer."
  exit 1
fi

"$PYTHON_BIN" --version

if ! "$PYTHON_BIN" -c "import venv, ensurepip" >/dev/null 2>&1; then
  echo "ERREUR : le module venv ou ensurepip manque."
  echo "Installe : sudo apt install python3.12-venv"
  exit 1
fi

# Recréer le venv s'il utilise une autre version de Python
if [ -x ".venv/bin/python" ] &&
   ! .venv/bin/python -c "import sys; exit(0 if sys.version_info[:2] == (3,12) else 1)"; then
  echo "L'environnement existant n'utilise pas Python 3.12."
  echo "Suppression de l'ancien environnement .venv ..."
  rm -rf .venv
fi

if [ ! -x ".venv/bin/python" ]; then
  echo "Creation de l'environnement virtuel .venv avec Python 3.12 ..."
  "$PYTHON_BIN" -m venv .venv
fi

# shellcheck disable=SC1091
source .venv/bin/activate
mkdir -p "$HOME/.cache/pip-tmp"
export TMPDIR="$HOME/.cache/pip-tmp"


python -m pip install --upgrade pip

# 3. Version de PyTorch (le pilote NVIDIA reste installe cote Windows)
echo
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
  echo "GPU NVIDIA visible depuis WSL :"
  nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
  DEFAULT=1
else
  echo "nvidia-smi ne repond pas dans WSL : aucun GPU NVIDIA utilisable pour l'instant."
  echo "(Si tu as une carte NVIDIA : mets a jour le pilote NVIDIA cote Windows, puis 'wsl --shutdown'.)"
  DEFAULT=3
fi
echo
echo "Quelle version de PyTorch installer ?"
echo "   1 = Carte NVIDIA courante : GTX 16xx, RTX 20, 30 ou 40 (dont RTX 2050)"
echo "   2 = Carte NVIDIA tres recente : RTX 50"
echo "   3 = Pas de NVIDIA, ou je ne sais pas : processeur, plus lent"
read -r -p "Ton choix (1, 2 ou 3) [${DEFAULT}] : " CHOICE
CHOICE="${CHOICE:-$DEFAULT}"
case "$CHOICE" in
  3) INDEX="cpu" ;;
  2) INDEX="cu128" ;;
  *) INDEX="cu126" ;;
esac

echo
echo "Installation de PyTorch depuis https://download.pytorch.org/whl/${INDEX}"
echo "Gros telechargement, plusieurs minutes, ne ferme pas le terminal."
pip install torch --index-url "https://download.pytorch.org/whl/${INDEX}"

# 4. Le reste
echo
echo "Installation des autres bibliotheques ..."
pip install -r requirements.txt

# 5. Verification
echo
python -X utf8 verifier_installation.py || true
echo
echo "Installation terminee. Etape suivante : bash lancer_jupyter.sh"
