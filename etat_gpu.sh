#!/usr/bin/env bash
# Montre la memoire video et les noyaux Jupyter qui tournent encore.
#   bash etat_gpu.sh              : affiche l'etat
#   bash etat_gpu.sh --liberer    : arrete les noyaux Jupyter de ce projet (libere la memoire video)
cd "$(dirname "$(readlink -f "$0")")" || exit 1
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv
else
  echo "nvidia-smi introuvable dans WSL (pilote NVIDIA cote Windows manquant ou ancien)."
fi
echo
echo "Noyaux Jupyter (ipykernel) de ce projet qui tournent :"
PIDS=$(pgrep -f "$PWD/.venv/bin/python.*ipykernel_launcher" || true)
if [ -z "$PIDS" ]; then
  echo "  aucun"
else
  for p in $PIDS; do echo "  PID $p"; done
  if [ "$1" = "--liberer" ]; then
    echo
    echo "Arret de ces noyaux ..."
    # shellcheck disable=SC2086
    kill $PIDS 2>/dev/null
    sleep 3
    # shellcheck disable=SC2086
    kill -9 $PIDS 2>/dev/null || true
    echo "Fait. Memoire video apres arret :"
    nvidia-smi --query-gpu=memory.used,memory.total --format=csv 2>/dev/null || true
  else
    echo
    echo "Pour les arreter : bash etat_gpu.sh --liberer"
    echo "(ils gardent le modele en memoire video tant que tu ne les arretes pas)"
  fi
fi
