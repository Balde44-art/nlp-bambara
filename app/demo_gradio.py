"""Interface de traduction fr <-> bambara (Gradio).



Prerequis : avoir fait le notebook 05 (dossier models/nllb-fr-bam-lora). Lancement :



    bash lancer_interface.sh        (ou : python app/demo_gradio.py)



Puis ouvrir http://127.0.0.1:7860 dans le navigateur Windows (WSL n'ouvre pas de navigateur tout seul).

Sur une carte de moins de 6 Go, un seul modele est charge (mode economique).

"""



import re

import sys

from pathlib import Path



sys.path.insert(0, str(Path(__file__).resolve().parent.parent))



import torch

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer



ROOT = Path(__file__).resolve().parent.parent

ADAPTER_DIR = ROOT / "models" / "nllb-fr-bam-lora"

MODEL_NAME = "facebook/nllb-200-distilled-600M"

FR, BAM = "fra_Latn", "bam_Latn"

MAX_CHARS = 2000



device = "cuda" if torch.cuda.is_available() else "cpu"

vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3 if device == "cuda" else 0.0

LOW_VRAM = device == "cuda" and vram_gb < 6
BATCH_SIZE = 1 if LOW_VRAM else 8

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)





def _finish(m):

    if device == "cuda":

        m = m.half()   # fp16 d'abord sur le processeur : evite un pic de memoire video

    return m.to(device).eval()





has_adapter = ADAPTER_DIR.exists()

if has_adapter:

    from peft import PeftModel



    peft_model = PeftModel.from_pretrained(AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME), str(ADAPTER_DIR))

    if LOW_VRAM:

        # Un seul modele en memoire : NLLB de base = ce modele avec l'adaptateur desactive.

        model_fr_bam = _finish(peft_model)

        model_base = None

    else:

        model_fr_bam = _finish(peft_model.merge_and_unload())

        model_base = _finish(AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME))

    FR_BAM_LABEL = "Français → Bambara (modèle fine-tuné)"

else:

    print(f"Adaptateur introuvable ({ADAPTER_DIR}) : fr -> bam utilisera NLLB de base.")

    model_base = _finish(AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME))

    model_fr_bam = model_base

    FR_BAM_LABEL = "Français → Bambara (NLLB de base)"



DIRECTIONS = {

    FR_BAM_LABEL: (FR, BAM, model_fr_bam),

    "Bambara → Français (NLLB de base)": (BAM, FR, model_base),

}





def translate(model, texts, src, tgt, num_beams=4, max_new_tokens=192):

    tokenizer.src_lang = src

    enc = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=256).to(device)

    with torch.no_grad():

        out = model.generate(

            **enc,

            forced_bos_token_id=tokenizer.convert_tokens_to_ids(tgt),

            num_beams=num_beams,

            max_new_tokens=max_new_tokens,

        )

    return tokenizer.batch_decode(out, skip_special_tokens=True)





def split_sentences(text):

    parts = re.split(r"(?<=[.!?])\s+|\n+", text.strip())

    return [p.strip() for p in parts if p.strip()]





def run(text, direction, beams):

    text = (text or "").strip()

    if not text:

        return ""

    if len(text) > MAX_CHARS:

        return f"Texte trop long ({len(text)} caractères, maximum {MAX_CHARS})."

    src, tgt, model = DIRECTIONS[direction]

    outputs = []

    sentences = split_sentences(text)

    for i in range(0, len(sentences), BATCH_SIZE):

        batch = sentences[i:i + BATCH_SIZE]

        if model is None:  # mode economique : adaptateur desactive pour obtenir NLLB de base

            with model_fr_bam.disable_adapter():

                outputs += translate(model_fr_bam, batch, src, tgt, num_beams=int(beams))

        else:

            outputs += translate(model, batch, src, tgt, num_beams=int(beams))

    return " ".join(outputs)





import time



import gradio as gr



# ---- Interface visuelle JALIKUMA ----
# Palette principale demandée : violet foncé #46063A
PALETTE = "violet_fonce"

PALETTES = {
    "violet_fonce": ("#46063A", "#6B245B", "violet"),
    "violet": ("#7C3AED", "#2563EB", "violet"),
    "orange": ("#F97316", "#DC2626", "orange"),
    "vert": ("#059669", "#0891B2", "emerald"),
    "bleu": ("#2563EB", "#0EA5E9", "blue"),
    "rose": ("#DB2777", "#7C3AED", "pink"),
}
C1, C2, HUE = PALETTES[PALETTE]
GR_MAJOR = int(gr.__version__.split(".")[0])
LABEL_FR_BAM, LABEL_BAM_FR = list(DIRECTIONS)[0], list(DIRECTIONS)[1]

if PALETTE == "violet_fonce":
    HUE = gr.themes.Color(
        c50="#F8F4F7", c100="#F0E6ED", c200="#DFC9D8",
        c300="#C59DB9", c400="#966487", c500="#46063A",
        c600="#3F0534", c700="#38052E", c800="#300427",
        c900="#260320", c950="#1D0318",
    )
THEME = gr.themes.Soft(primary_hue=HUE, neutral_hue="slate", radius_size="lg")

CSS = r"""
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;650;700;750&display=swap');
:root {
  --plum: #46063A; --plum2: #6B245B; --ink: #211724; --muted: #786D7A;
  --font-ui: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
}
html, body, .gradio-container, .gradio-container button,
.gradio-container input, .gradio-container textarea,
.gradio-container select, .gradio-container label,
.gradio-container p, .gradio-container h1,
.gradio-container h2, .gradio-container h3 {
  font-family: var(--font-ui) !important;
}
.gradio-container { max-width: 1240px !important; margin: 0 auto !important; background: #FBF9FC !important; }
footer { display: none !important; }
#topbar { display:flex; align-items:center; justify-content:space-between; gap:18px; padding:12px 4px 18px; border-bottom:1px solid #EDE4EC; margin-bottom:22px; }
.brand { display:flex; align-items:center; gap:12px; color:var(--plum); font-weight:650; font-size:1.22rem; }
.brand-mark { width:42px; height:42px; display:grid; place-items:center; color:white; background:var(--plum); border-radius:13px; font-size:1.2rem; }
.brand small { display:block; color:#817582; font-size:.72rem; font-weight:500; margin-top:2px; }
.nav { color:#625563; font-size:.9rem; }
#intro { text-align:center; padding:22px 12px 26px; }
#intro h1 { color:#211724; font-size:clamp(1.8rem,3vw,2.65rem); font-weight:650; letter-spacing:-.035em; line-height:1.16; margin:0 0 8px; }
#intro h1 span { color:var(--plum); }
#intro p { color:#766B78; font-size:1.02rem; font-weight:400; line-height:1.65; margin:0; }
.mali-line { display:flex; justify-content:center; gap:7px; margin:17px auto 0; }
.mali-line i { display:block; height:4px; width:42px; border-radius:9px; }
.mali-line i:nth-child(1){background:#14B53A}.mali-line i:nth-child(2){background:#FCD116}.mali-line i:nth-child(3){background:#CE1126}
#workspace { background:white; border:1px solid #E9DFE9; border-radius:22px; padding:20px !important; box-shadow:0 12px 34px rgba(70,6,58,.07); }
.lang-label { color:#302433; font-weight:600; font-size:1rem; margin:0 0 8px; }
.lang-label span { color:var(--plum); margin-right:8px; }
textarea { border-radius:14px !important; border:1px solid #E4D9E5 !important; background:#FEFDFE !important; min-height:210px !important; font-size:1rem !important; line-height:1.6 !important; }
#output textarea { background:#FBF8FC !important; }
#translate-btn { background:var(--plum) !important; color:white !important; border:0 !important; border-radius:999px !important; font-weight:600 !important; letter-spacing:-.01em; min-height:48px; box-shadow:0 7px 16px rgba(70,6,58,.18); }
#translate-btn:hover { background:#5B124D !important; }
#swap-btn { border:1px solid #E4D8E5 !important; color:var(--plum) !important; border-radius:999px !important; }
#secondary-btn { border-radius:999px !important; }
#status { color:#766B78; font-size:.86rem; text-align:center; }
#counter { color:#8A7D8D; font-size:.78rem; text-align:right; }
#tool-tabs { margin-top:20px; }
#history-panel { background:white; border:1px solid #E9DFE9; border-radius:18px; padding:16px; }
#history-panel h3 { color:#302433; margin:0 0 8px; font-size:1rem; }
.history-item { padding:10px 12px; border-radius:10px; background:#F7F1F7; margin:7px 0; color:#453747; }
#footer-note { text-align:center; color:#8A7D8D; font-size:.78rem; line-height:1.6; padding:22px 10px 8px; }
#footer-note strong { color:var(--plum); }
@media (max-width: 700px) { #topbar { align-items:flex-start; flex-direction:column; } #workspace { padding:12px !important; } textarea { min-height:170px !important; } }
"""

HERO = """
<div id="topbar">
  <div class="brand"><div class="brand-mark">JK</div><div>JALIKUMA<small>La parole sans frontières</small></div></div>
  <div class="nav">Accueil　 ·　 Traduction　 ·　 Notre vision</div>
  <div class="nav">🇲🇱　Mali</div>
</div>
<div id="intro">
  <h1>JALIKUMA : <span>la parole</span> sans frontières</h1>
  <p>Traduisez simplement entre le français et le bambara</p>
  <div class="mali-line"><i></i><i></i><i></i></div>
</div>
"""

FOOTER = """
<div id="footer-note">
  <strong>JALIKUMA</strong> · Traduction intelligente · Français ↔ Bambara · Mali<br>
  Les traductions automatiques peuvent contenir des erreurs. Faites-les vérifier par un locuteur.<br>
  Modèle : NLLB-200 (Meta) · Données : MAFAND-MT (Masakhane) · Licence CC BY-NC-4.0
</div>
"""

EXAMPLES = [
    ["Bonjour, comment allez-vous aujourd'hui ?", LABEL_FR_BAM, 2 if LOW_VRAM else 4],
    ["Le ministre a annoncé la réouverture des écoles lundi prochain.", LABEL_FR_BAM, 2 if LOW_VRAM else 4],
    ["Irrégularités financières.", LABEL_FR_BAM, 2 if LOW_VRAM else 4],
    ["I ni sogoma, i ka kɛnɛ wa ?", LABEL_BAM_FR, 2 if LOW_VRAM else 4],
]

def counter(text):
    return f"{len(text or '')} / {MAX_CHARS} caractères"

def render_history(items):
    if not items:
        return "Aucune traduction récente pour le moment."
    rows = []
    for item in reversed(items[-5:]):
        source = item[0].replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        target = item[1].replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        rows.append(f'<div class="history-item"><b>{source}</b><br><span style="color:#46063A">↳ {target}</span></div>')
    return "".join(rows)

def translate_ui(text, direction, beams, history):
    text = (text or "").strip()
    history = list(history or [])
    if not text:
        return "", "Saisissez un texte pour commencer.", render_history(history), history
    t0 = time.time()
    out = run(text, direction, beams)
    if out.startswith("Texte trop long"):
        return out, "La limite de caractères est dépassée.", render_history(history), history
    n = len(split_sentences(text))
    history.append((text, out))
    history = history[-20:]
    return out, f"Traduction terminée · {time.time() - t0:.1f} s · {n} phrase(s)", render_history(history), history

def swap(text_in, text_out, direction):
    new_dir = LABEL_BAM_FR if direction == LABEL_FR_BAM else LABEL_FR_BAM
    return text_out or text_in, "", new_dir, ""

blocks_kwargs = {"title": "JALIKUMA — Traduction français ↔ bambara"}
if GR_MAJOR < 6:
    blocks_kwargs.update(theme=THEME, css=CSS)

with gr.Blocks(**blocks_kwargs) as demo:
    gr.HTML(HERO)
    history_state = gr.State([])
    with gr.Column(elem_id="workspace"):
        with gr.Row(equal_height=True):
            with gr.Column(scale=1):
                gr.Markdown('<div class="lang-label"><span>🇫🇷</span> Français</div>')
                direction = gr.Radio(choices=[LABEL_FR_BAM, LABEL_BAM_FR], value=LABEL_FR_BAM,
                                     label="Sens de traduction", interactive=True)
                text_in = gr.Textbox(label="Votre texte", lines=8,
                                     placeholder="Saisissez votre texte en français…", max_lines=12)
                count = gr.Markdown(counter(""), elem_id="counter")
            with gr.Column(scale=1, elem_id="output"):
                gr.Markdown('<div class="lang-label"><span>🇲🇱</span> Bambara / Français</div>')
                text_out = gr.Textbox(label="Résultat de la traduction", lines=8,
                                      placeholder="La traduction apparaîtra ici…", interactive=False)
                gr.Markdown("La traduction générée peut nécessiter une vérification.")
        with gr.Row():
            swap_btn = gr.Button("⇄ Inverser les langues", elem_id="swap-btn", scale=1)
            btn = gr.Button("✦　Traduire", variant="primary", elem_id="translate-btn", scale=2)
            clear_btn = gr.Button("Effacer", elem_id="secondary-btn", scale=1)
        info = gr.Markdown("", elem_id="status")
        with gr.Accordion("Paramètres avancés", open=False):
            beams = gr.Slider(1, 4, value=(2 if LOW_VRAM else 4), step=1,
                              label="Qualité (faisceaux) — plus élevé = plus lent")
    with gr.Tabs(elem_id="tool-tabs"):
        with gr.Tab("Texte"):
            gr.Markdown("Saisissez votre texte, choisissez le sens, puis lancez la traduction.")
            gr.Examples(examples=EXAMPLES, inputs=[text_in, direction, beams],
                        label="Exemples pour essayer")
        with gr.Tab("Historique"):
            gr.Markdown("### Traductions récentes")
            history_view = gr.HTML('<div id="history-panel">Aucune traduction récente pour le moment.</div>')
    gr.HTML(FOOTER)

    text_in.change(counter, text_in, count)
    btn.click(translate_ui, [text_in, direction, beams, history_state],
              [text_out, info, history_view, history_state])
    text_in.submit(translate_ui, [text_in, direction, beams, history_state],
                   [text_out, info, history_view, history_state])
    swap_btn.click(swap, [text_in, text_out, direction], [text_in, text_out, direction, info])
    clear_btn.click(lambda: ("", "", ""), None, [text_in, text_out, info])

if __name__ == "__main__":
    print("Modèles chargés. Ouvre http://127.0.0.1:7860 dans le navigateur Windows.")
    LAUNCH = dict(server_name="127.0.0.1", server_port=7860, inbrowser=False)
    if GR_MAJOR >= 6:
        try:
            demo.launch(theme=THEME, css=CSS, **LAUNCH)
        except TypeError:
            demo.launch(**LAUNCH)
    else:
        demo.launch(**LAUNCH)
