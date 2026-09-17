"""Gradio UI for the offensive language detection pipeline.

Design notes: a neutral sage/paper background with ink text, IBM Plex Sans
for interface text and IBM Plex Mono for data (timestamps, flagged words) --
a technical pairing that fits an audio-processing tool rather than a
generic SaaS look. Color carries meaning, not decoration: deep pine for
primary actions and results, burnt orange only on flagged content.
"""
import gradio as gr

from pipeline import run_pipeline

WORK_DIR = "workdir"

INK = "#14181C"
BACKGROUND = "#F5F6F4"
PANEL = "#FFFFFF"
BORDER = "#DCDFDB"
PRIMARY = "#2B5D50"
PRIMARY_HOVER = "#204A3F"
FLAGGED = "#C1440E"

CUSTOM_CSS = f"""
.gradio-container {{ max-width: 960px !important; margin: 0 auto !important; }}
#header-block h1 {{ font-weight: 600; letter-spacing: -0.01em; margin-bottom: 4px; }}
#header-block p {{ color: #4B5049; font-size: 15px; line-height: 1.5; max-width: 640px; }}
.output-panel {{ border-left: 3px solid {PRIMARY} !important; padding-left: 16px !important; }}
.flagged-panel {{ border-left: 3px solid {FLAGGED} !important; padding-left: 16px !important; margin-top: 8px; }}
footer {{ visibility: hidden }}
"""

theme = gr.themes.Base(
    font=[gr.themes.GoogleFont("IBM Plex Sans"), "ui-sans-serif", "sans-serif"],
    font_mono=[gr.themes.GoogleFont("IBM Plex Mono"), "ui-monospace", "monospace"],
).set(
    body_background_fill=BACKGROUND,
    body_text_color=INK,
    block_background_fill=PANEL,
    block_border_color=BORDER,
    block_label_text_color=INK,
    block_title_text_color=INK,
    button_primary_background_fill=PRIMARY,
    button_primary_background_fill_hover=PRIMARY_HOVER,
    button_primary_text_color="#FFFFFF",
    input_background_fill=PANEL,
    border_color_primary=BORDER,
)


def _normalize(word: str) -> str:
    """Match the normalization used inside pipeline.censorship.match_toxic_timestamps."""
    return word.strip().lower().strip(".,!?\"'")


def _highlighted_tokens(text: str, toxic_words):
    """Build (token, label) pairs for gr.HighlightedText, flagging toxic words."""
    toxic_set = set(toxic_words)
    return [
        (word + " ", "flagged" if _normalize(word) in toxic_set else None)
        for word in text.split()
    ]


def process_video(video_path):
    if not video_path:
        raise gr.Error("Upload a video first.")

    result = run_pipeline(video_path, work_dir=WORK_DIR)

    highlighted = _highlighted_tokens(result["transcript"], result["toxic_words"])

    if result["toxic_timestamps"]:
        rows = [
            [_normalize(w["word"]), round(w["start_ms"] / 1000, 2), round(w["end_ms"] / 1000, 2)]
            for w in result["toxic_timestamps"]
        ]
    else:
        rows = [["—", "—", "—"]]

    words_found = len(result["toxic_words"])
    status = (
        f"Found and censored {words_found} offensive word(s)."
        if words_found
        else "No offensive words detected in this clip."
    )

    return result["censored_video_path"], highlighted, rows, status


with gr.Blocks(title="Offensive Language Detection", theme=theme, css=CUSTOM_CSS) as demo:
    gr.Markdown(
        "# Offensive language detection\n"
        "Upload a clip and it censors offensive speech automatically — flagged "
        "words get beeped out, and you can see exactly what was caught and why.",
        elem_id="header-block",
    )

    with gr.Row():
        with gr.Column():
            video_in = gr.Video(label="Input video")
            run_btn = gr.Button("Censor video", variant="primary")
        with gr.Column(elem_classes=["output-panel"]):
            video_out = gr.Video(label="Censored output")
            status = gr.Textbox(label="Status", interactive=False)

    with gr.Column(elem_classes=["flagged-panel"]):
        transcript_out = gr.HighlightedText(
            label="Transcript",
            color_map={"flagged": FLAGGED},
        )
        timestamps_out = gr.Dataframe(
            headers=["Word", "Start (s)", "End (s)"],
            label="Flagged words",
        )

    run_btn.click(
        fn=process_video,
        inputs=video_in,
        outputs=[video_out, transcript_out, timestamps_out, status],
    )

if __name__ == "__main__":
    demo.launch()