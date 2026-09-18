"""Streamlit UI for the offensive language detection pipeline.

Same design language as the earlier Gradio version (sage background, pine
green for actions/results, burnt orange only on flagged content, IBM Plex
for type) so the visual identity carries over even though the framework
changed.
"""
import os

import streamlit as st

from pipeline import run_pipeline

WORK_DIR = "workdir"

INK = "#14181C"
BACKGROUND = "#F5F6F4"
PRIMARY = "#2B5D50"
PRIMARY_HOVER = "#204A3F"
FLAGGED = "#C1440E"

st.set_page_config(page_title="Offensive Language Detection", page_icon="🔇", layout="centered")

st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;600&family=IBM+Plex+Mono&display=swap');
    html, body, [class*="css"] {{ font-family: 'IBM Plex Sans', sans-serif; color: {INK}; }}
    .stApp {{ background-color: {BACKGROUND}; }}
    div.stButton > button:first-child {{ background-color: {PRIMARY}; color: white; border: none; }}
    div.stButton > button:first-child:hover {{ background-color: {PRIMARY_HOVER}; color: white; }}
    .output-panel {{ border-left: 3px solid {PRIMARY}; padding-left: 16px; margin-top: 12px; }}
    .flagged-panel {{ border-left: 3px solid {FLAGGED}; padding-left: 16px; margin-top: 12px; }}
    .flagged-word {{ background-color: {FLAGGED}; color: white; padding: 1px 5px; border-radius: 3px; }}
    [data-testid="stDataFrame"] {{ font-family: 'IBM Plex Mono', monospace; }}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Offensive language detection")
st.write(
    "Upload a clip and it censors offensive speech automatically — flagged "
    "words get beeped out, and you can see exactly what was caught and why."
)


def _normalize(word: str) -> str:
    """Match the normalization used inside pipeline.censorship.match_toxic_timestamps."""
    return word.strip().lower().strip(".,!?\"'")


def _highlighted_html(text: str, toxic_words) -> str:
    """Render the transcript as HTML with flagged words wrapped in a highlight span."""
    toxic_set = set(toxic_words)
    parts = [
        f'<span class="flagged-word">{word}</span>' if _normalize(word) in toxic_set else word
        for word in text.split()
    ]
    return " ".join(parts)


uploaded = st.file_uploader("Input video", type=["mp4", "mov", "mkv", "avi"])
run_clicked = st.button("Censor video", type="primary", disabled=uploaded is None)

if run_clicked and uploaded is not None:
    with st.spinner("Processing — can take a minute or two on CPU."):
        os.makedirs(WORK_DIR, exist_ok=True)
        input_path = os.path.join(WORK_DIR, f"input_{uploaded.name}")
        with open(input_path, "wb") as f:
            f.write(uploaded.getbuffer())

        result = run_pipeline(input_path, work_dir=WORK_DIR)

    words_found = len(result["toxic_words"])
    status = (
        f"Found and censored {words_found} offensive word(s)."
        if words_found
        else "No offensive words detected in this clip."
    )

    st.markdown('<div class="output-panel">', unsafe_allow_html=True)
    st.subheader("Censored output")
    st.video(result["censored_video_path"])
    st.caption(status)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="flagged-panel">', unsafe_allow_html=True)
    st.subheader("Transcript")
    st.markdown(_highlighted_html(result["transcript"], result["toxic_words"]), unsafe_allow_html=True)

    if result["toxic_timestamps"]:
        st.subheader("Flagged words")
        rows = [
            {
                "Word": _normalize(w["word"]),
                "Start (s)": round(w["start_ms"] / 1000, 2),
                "End (s)": round(w["end_ms"] / 1000, 2),
            }
            for w in result["toxic_timestamps"]
        ]
        st.dataframe(rows, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)