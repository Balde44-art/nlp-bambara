# nlp-bambara — traduction et détection de langue français ↔ bambara (WSL, carte de 4 Go)

Projet d'apprentissage du NLP :

- un **détecteur de langue** (français, anglais, bambara et six langues voisines) ;
- un **traducteur français ↔ bambara** basé sur NLLB-200, avec fine-tuning LoRA pour le sens français → bambara ;
- une **interface web** pour utiliser le traducteur.

Tout tourne **dans WSL2 (Ubuntu)**, dans des notebooks Jupyter commentés. Cette version est réglée pour une **carte NVIDIA de 4 Go** (par exemple RTX 2050).

## Aperçu

<!--
  POUR AJOUTER TON IMAGE (capture de l'interface) :
  1. Envoie l'image sur ton Google Drive.
  2. Clic droit sur l'image > Partager > Accès général : « Toute personne disposant du lien ».
  3. Clique sur « Copier le lien » et colle-le à la place de REMPLACE_PAR_TON_LIEN_DRIVE ci-dessous.

  OPTION 1 (fiable) : un lien cliquable, déjà en place ci-dessous.

  OPTION 2 (image affichée directement dans la page) : Google Drive bloque parfois l'affichage
  dans GitHub. Si tu veux essayer, remplace ID_DU_FICHIER par la partie du lien située entre
  /d/ et /view, puis copie la ligne suivante EN DEHORS de ce commentaire (sous le lien cliquable) :
  ![Aperçu de l'interface](https://drive.google.com/uc?export=view&id=ID_DU_FICHIER)

  OPTION 3 (la plus fiable pour afficher l'image) : mets le fichier dans le dossier docs/ du projet
  (par exemple docs/apercu.png) et utilise : ![Aperçu de l'interface](docs/apercu.png)
  📷 **[Voir l'aperçu de l'interface](REMPLACE_PAR_TON_LIEN_DRIVE)**
-->

![Mon Schémas](nlp-bambara-wsl\img\immg.png)

---

## 1. Les notebooks

| N° | Notebook | À quoi il sert | GPU NVIDIA ? |
|---|---|---|---|
| 01 | `01_exploration_donnees` | Télécharger et nettoyer le corpus | non |
| 02 | `02_detection_langue` | Détecteur de langue (9 langues) | non |
| 03 | `03_traduction_nllb_baseline` | Mesurer NLLB-200 tel quel | recommandé |
| 05 | `05_finetuning` | Fine-tuning LoRA français → bambara | **indispensable** |
| 04 | `04_evaluation` | Score final et analyse d'erreurs (après le 05) | recommandé |
| 06 | `06_interface_gradio` | Interface web de traduction | recommandé |

**Ordre conseillé : 01 → 02 → 03 → 05 → 04 → 06.**

---

## 2. Ce que fait le projet pour tenir dans 4 Go

Le mode **économe** (`LOW_VRAM`) s'active tout seul quand la carte a moins de 6 Go :

- le modèle de base, gelé pendant l'entraînement, est gardé en **demi-précision** (1,2 Go au lieu de 2,4 Go) ;
- le **gradient checkpointing** recalcule certains résultats au lieu de les stocker ;
- **1 phrase à la fois**, accumulée sur 16 (le lot effectif reste 16) ;
- dans tous les notebooks, le modèle est converti en demi-précision **avant** d'être envoyé sur la carte (pas de pic de mémoire) ;
- avant l'évaluation du notebook 05, le matériel d'entraînement est libéré ;
- l'interface (notebook 06 et `app/demo_gradio.py`) ne charge qu'**un seul** modèle.

**Ce n'est pas garanti.** Ces réglages n'ont pas pu être essayés sur une vraie carte de 4 Go. Windows garde en plus une partie de la mémoire vidéo pour l'affichage. Voir la section 7 si une erreur `CUDA out of memory` apparaît.

---

## 3. Prérequis

- **WSL2 + Ubuntu** (`wsl --install -d Ubuntu` dans PowerShell). Vérifie avec `wsl -l -v` : la colonne VERSION doit afficher `2`.
- **Pilote NVIDIA récent installé sous Windows uniquement.** Ne rien installer de NVIDIA dans Ubuntu. Dans Ubuntu, `nvidia-smi` doit afficher ta carte.
- **Python 3.10 ou plus** (Ubuntu 24.04 : 3.12, la version testée).
- **Disque** : environ 12 Go libres. **Mémoire vive de WSL** : environ 6 Go suffisent pour charger le modèle ; plus est plus confortable (voir la section 7, ligne `Killed`).

---

## 4. Installation (une seule fois)

Dans le terminal **Ubuntu**.

**Étape 1 — Paquets système :**

```bash
sudo apt update && sudo apt install -y python3 python3-venv python3-pip unzip zip
```

**Étape 2 — Mettre le projet dans le système de fichiers Linux** (pas sur `/mnt/c`, trop lent). Remplace `<ton_nom>` par ton nom d'utilisateur Windows :

```bash
cp /mnt/c/Users/<ton_nom>/Downloads/nlp-bambara-wsl-4go.zip ~/
cd ~
unzip nlp-bambara-wsl-4go.zip
cd nlp-bambara-wsl
```

**Étape 3 — Récupérer tes anciennes données** (seulement si tu as déjà fait les notebooks 01 à 03 dans l'ancien dossier) :

```bash
OLD="/mnt/c/Users/<ton_nom>/Desktop/nlp-bambara-wsl/nlp-bambara-wsl"
cp -r "$OLD/data" "$OLD/results" ~/nlp-bambara-wsl/
mkdir -p ~/nlp-bambara-wsl/models
cp "$OLD"/models/langid.joblib ~/nlp-bambara-wsl/models/ 2>/dev/null || true
```

On ne copie pas `models/checkpoints` : ce sont des sauvegardes d'un entraînement qui a échoué. Ne copie pas non plus `.venv` : il est lié à son dossier, on le recrée à l'étape suivante.

**Étape 4 — Installer :**

```bash
bash installer.sh
```

Le script crée `.venv` et te demande la version de PyTorch : choisis **1** pour une RTX 2050 (proposé par défaut si la carte est vue). Compte plusieurs minutes, ne ferme pas le terminal. À la fin, la vérification doit afficher `Tout est pret`, `GPU utilisable : ... RTX 2050` et la mémoire vidéo libre.

---

## 5. Utilisation, pas à pas

Toutes les commandes se tapent depuis `~/nlp-bambara-wsl`, dans Ubuntu.

| Commande | Effet |
|---|---|
| `bash verifier_installation.sh` | Vérifie Python, bibliothèques, GPU et mémoire vidéo libre |
| `bash lancer_tests.sh` | Tests du détecteur de langue (2 tests attendus) |
| `bash etat_gpu.sh` | Montre la mémoire vidéo et les noyaux Jupyter qui tournent encore |
| `bash etat_gpu.sh --liberer` | Arrête ces noyaux : libère la mémoire vidéo |
| `bash lancer_jupyter.sh` | Démarre JupyterLab : copie l'adresse `http://127.0.0.1:8888/lab?token=...` dans le navigateur **Windows** |
| `bash lancer_interface.sh` | Interface de traduction sur `http://127.0.0.1:7860` (après le notebook 05, et JupyterLab fermé) |

**Déroulé :**

1. `bash verifier_installation.sh` puis `bash lancer_tests.sh`.
2. `bash lancer_jupyter.sh`, ouvrir l'adresse dans Windows, dossier `notebooks`.
3. **01** puis **02** : exécute les cellules une par une (`Maj + Entrée`).
4. **03** : mesure NLLB tel quel. Quand tu as fini, ferme-le : menu **Kernel → Shut Down Kernel**.
5. **05**, avec un **essai rapide d'abord** : dans la cellule « 5. Régler les paramètres », remplace temporairement `EPOCHS = 4` par `EPOCHS = 1` et `TRAIN_SUBSET = None if device == "cuda" else 64` par `TRAIN_SUBSET = 200`. Exécute jusqu'à la fin de l'entraînement. Si aucune erreur `CUDA out of memory` n'apparaît, **remets les valeurs d'origine** (`EPOCHS = 4`, `TRAIN_SUBSET = None if device == "cuda" else 64`), redémarre le noyau (**Kernel → Restart Kernel**) et lance le vrai entraînement.
6. Pendant l'entraînement, regarde dans un second terminal `nvidia-smi` : la mémoire doit rester sous 4096 MiB.
7. **04** : score final et analyse d'erreurs. Ferme le noyau du 05 d'abord.
8. **06**, ou `bash lancer_interface.sh` après avoir fermé JupyterLab (`Ctrl+C` deux fois).

**Règle d'or avec 4 Go : un seul notebook chargé à la fois.** Chaque noyau Jupyter garde son modèle en mémoire vidéo tant qu'il n'est pas arrêté.

**Reprendre un autre jour :**

```bash
cd ~/nlp-bambara-wsl
bash lancer_jupyter.sh
```

---

## 6. Données supplémentaires et plan B Google Colab

**Données supplémentaires (facultatif).** Place un fichier `donnees_supplementaires.tsv` (colonnes `fr` et `bam`, séparées par une tabulation) dans `data/processed/`. Le notebook 05 (cellule « 2 bis ») l'ajoute à l'entraînement en nettoyant les paires et en retirant celles qui se trouvent déjà dans la validation ou le test. Sans ce fichier, rien ne change.

**Plan B : faire le notebook 05 sur Colab.** Si l'entraînement ne tient pas sur ta carte :

1. Dans WSL, depuis le projet : `zip -r colab_05.zip src data/processed results/metrics.csv results/predictions_nllb_baseline.tsv`
2. Mets `colab_05.zip` (et éventuellement `donnees_supplementaires.tsv`) dans un dossier **`nlp-bambara`** de ton Google Drive.
3. Sur Colab : importe `notebooks/05_finetuning.ipynb`, choisis **T4 GPU**, et exécute toutes les cellules dans l'ordre. La cellule 0 monte ton Drive et prépare tout. L'adaptateur est copié dans le Drive.
4. Télécharge `nllb-fr-bam-lora` depuis le Drive et place-le dans `models/` ici. Les notebooks 04 et 06 fonctionnent ensuite normalement.

---

## 7. Problèmes courants

| Symptôme | Cause probable | Solution |
|---|---|---|
| `CUDA out of memory` | Mémoire vidéo prise par un noyau oublié ou un autre programme | `bash etat_gpu.sh --liberer`, ferme navigateur/jeux, relance, et refais le notebook depuis la première cellule |
| `CUDA out of memory` malgré tout | 4 Go trop justes | Dans la cellule des paramètres du 05 : `MAX_LEN = 96`, puis `LORA_R = 16` et `LORA_ALPHA = 32`. Sinon, plan B Colab (section 6) |
| La vérification dit « Memoire video libre : < 3 Go » | Un noyau Jupyter ou une appli garde la carte | `bash etat_gpu.sh`, puis `--liberer` |
| `Killed` pendant le chargement du modèle | Mémoire vive de WSL insuffisante | Crée `C:\Users\<toi>\.wslconfig` avec `[wsl2]` puis `memory=8GB` (seulement si ton PC a assez de RAM), puis `wsl --shutdown` dans PowerShell |
| `nvidia-smi` échoue dans WSL | Pilote Windows absent ou ancien, ou WSL 1 | Mets à jour le pilote NVIDIA côté Windows, `wsl --shutdown`, vérifie `wsl -l -v` (version 2) |
| `bad interpreter: /bin/bash^M` ou `\r: command not found` | Fins de ligne Windows dans un `.sh` | `sed -i 's/\r$//' *.sh` |
| `Permission denied` avec `./script.sh` | Droit d'exécution perdu | Utilise `bash script.sh`, ou `chmod +x *.sh` |
| `ensurepip is not available` | `python3-venv` absent | `sudo apt install -y python3-venv` |
| « PyTorch version PROCESSEUR » alors que tu as une NVIDIA | Mauvaise version de PyTorch | `rm -rf .venv`, puis `bash installer.sh`, choix 1 |
| « Dossier du projet introuvable » | Jupyter lancé depuis ailleurs | Utilise `bash lancer_jupyter.sh` |
| « ... introuvable : lance d'abord le notebook XX » | Notebook précédent non terminé | Exécute le notebook indiqué |
| Téléchargement du modèle interrompu | Connexion instable | Relance la cellule : il reprend |
| Le navigateur n'ouvre pas Jupyter ou Gradio | Redirection localhost | Copie l'adresse affichée dans le navigateur Windows ; en dernier recours `wsl --shutdown` puis relance |

---

## 8. Organisation du dossier

```
nlp-bambara-wsl/
├── installer.sh               installation (une seule fois)
├── lancer_jupyter.sh          ouvre les notebooks
├── lancer_interface.sh        ouvre l'interface de traduction
├── verifier_installation.sh   vérifie Python, bibliothèques, GPU, mémoire vidéo libre
├── lancer_tests.sh            tests du détecteur de langue
├── etat_gpu.sh                mémoire vidéo et noyaux Jupyter à arrêter
├── verifier_installation.py   utilisé par les scripts ci-dessus
├── requirements.txt
├── notebooks/                 les 6 notebooks
├── src/                       langid.py (utilisé par les tests) ; le dossier doit exister
├── app/demo_gradio.py         interface web (version script)
├── tests/
├── data/                      raw/ (téléchargé) et processed/ (train/val/test.tsv)
├── models/                    langid.joblib et nllb-fr-bam-lora (créés par les notebooks)
└── results/                   metrics.csv, prédictions, figures, errors_analysis.md
```

**Sauvegarde ce que tu veux garder** : `models/nllb-fr-bam-lora` est le fruit de l'entraînement, `results/` contient tes mesures. Depuis Windows, tes fichiers WSL sont visibles dans l'Explorateur via `\\wsl$\Ubuntu\home\<toi>\nlp-bambara-wsl`.

---

## 9. Licences et limites

- Le corpus **MAFAND-MT** (Masakhane) et le modèle **NLLB-200** (Meta) sont sous licence **CC BY-NC-4.0** : usage non commercial, avec attribution.
- Le corpus fr-bam est petit (environ 3 000 paires d'entraînement) et vient de la presse : le traducteur sera moins bon sur des conversations courantes.
- La traduction automatique est à vérifier par un locuteur avant tout usage important.

---

## 10. État de la vérification

Préparé dans un environnement sans GPU, sans WSL et sans accès à Hugging Face.

- **Vérifiés** : syntaxe des 6 scripts `.sh` (`bash -n`) et de toutes les cellules de code des 6 notebooks, tests du détecteur de langue, lancement de `verifier_installation.py`, réglages automatiques de la cellule des paramètres selon la mémoire vidéo (4, 8 et 15 Go), nettoyage et fusion des données supplémentaires sur des données fictives.
- **Non testés** : l'exécution réelle des scripts sous WSL, l'installation de PyTorch, le chargement de NLLB, l'entraînement en mode économe sur 4 Go, l'interface, et le plan B Colab.

Si un script ou une cellule échoue, copie le message d'erreur complet.
