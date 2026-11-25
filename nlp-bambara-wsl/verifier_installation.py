"""Verifie que l'installation est complete et dit si le GPU est utilisable.

Lancement (dans WSL) :
    bash verifier_installation.sh
"""

import importlib
import platform
import shutil
import sys
from pathlib import Path

OK, KO, WARN = "[OK]  ", "[ERREUR]", "[ATTENTION]"
problems = 0


def say(tag, text):
    print(f"{tag} {text}")


print("=== Verification de l'installation ===\n")

say(OK, f"Python {platform.python_version()} ({platform.system()} {platform.release()})")
if sys.version_info < (3, 10) or sys.version_info >= (3, 14):
    say(WARN, "Version de Python inhabituelle : Python 3.11 ou 3.12 est le choix le plus sur.")

# Bibliotheques
for module in ["transformers", "datasets", "sentencepiece", "sacrebleu", "sklearn", "pandas",
               "matplotlib", "joblib", "accelerate", "peft", "gradio", "jupyterlab"]:
    try:
        m = importlib.import_module(module)
        say(OK, f"{module} {getattr(m, '__version__', '')}".rstrip())
    except Exception as e:  # noqa: BLE001
        problems += 1
        say(KO, f"{module} : {type(e).__name__} : {e}")

# PyTorch et GPU
print()
try:
    import torch

    say(OK, f"PyTorch {torch.__version__} (CUDA compilee : {torch.version.cuda})")
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        say(OK, f"GPU utilisable : {torch.cuda.get_device_name(0)}, {props.total_memory / 1024**3:.1f} Go de memoire video")
        if props.total_memory / 1024**3 < 6:
            say(WARN, "Moins de 6 Go de memoire video : le notebook 05 passera en mode economique (fp16, lots minuscules). Pas garanti sur 4 Go : ferme tout ce qui utilise la carte.")
        free_b, total_b = torch.cuda.mem_get_info()
        say(OK if free_b / 1024**3 >= 3 else WARN,
            f"Memoire video libre : {free_b / 1024**3:.1f} Go sur {total_b / 1024**3:.1f} Go"
            + ("" if free_b / 1024**3 >= 3 else " -> trop peu : un noyau Jupyter ou un programme garde la carte (bash etat_gpu.sh)"))
        try:
            x = torch.randn(256, 256, device="cuda")
            _ = (x @ x).sum().item()
            say(OK, "Un petit calcul sur le GPU fonctionne.")
        except Exception as e:  # noqa: BLE001
            problems += 1
            say(KO, f"Le GPU est vu mais un calcul echoue : {e}")
    else:
        say(WARN, "Aucun GPU utilisable : tout tournera sur le processeur (notebooks 01, 02 : OK ; 03, 04, 06 : lents ; 05 : impossible en pratique).")
        if torch.version.cuda is None:
            print("            PyTorch est installe en version PROCESSEUR. Si tu as une carte NVIDIA,")
            print("            supprime le dossier .venv puis relance bash installer.sh et choisis l'option NVIDIA.")
except Exception as e:  # noqa: BLE001
    problems += 1
    say(KO, f"PyTorch : {type(e).__name__} : {e}")

# Espace disque
print()
free_gb = shutil.disk_usage(Path(__file__).resolve().parent).free / 1024**3
say(OK if free_gb >= 8 else WARN, f"Espace disque libre : {free_gb:.1f} Go (le modele NLLB demande environ 2,5 Go de plus au premier lancement)")

print()
if problems:
    say(KO, f"{problems} probleme(s) a corriger avant de continuer.")
else:
    say(OK, "Tout est pret. Lance bash lancer_jupyter.sh puis ouvre le notebook 01.")
