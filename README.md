# Offensive Language Detection

Automatically detects and censors offensive language in a video by beeping
out flagged words in the audio track.

**Pipeline**: video → extract audio → transcribe with word-level timestamps
(Whisper) → classify words for offensiveness (transformer) → generate a beep
over each flagged word's timestamp → merge the censored audio back into the
video.

## Status

Core pipeline: done. Streamlit UI: done. Public deployment: in progress.

## Project structure

```
app.py                     # Streamlit UI
pipeline/
  audio_extraction.py    # video -> audio (moviepy)
  transcription.py       # audio -> transcript + word timestamps (Whisper)
  toxicity_detection.py  # transcript -> flagged words + sentence scores
  censorship.py          # flagged words -> beeped audio -> merged video
  utils.py                # shared helpers
requirements.txt
packages.txt              # apt packages for Streamlit Community Cloud (ffmpeg)
```

## Setup

```bash
pip install -r requirements.txt
```

ffmpeg must also be installed on the system (`apt install ffmpeg` on
Debian/Ubuntu, `brew install ffmpeg` on macOS).

## Usage

Run the web UI locally:

```bash
streamlit run app.py
```

Or use the pipeline directly:

```python
from pipeline import run_pipeline

result = run_pipeline("my_video.mp4", work_dir="workdir")
print(result["transcript"])
print(result["toxic_words"])
print(result["censored_video_path"])
```

## Known limitations

- Offensiveness is scored word-by-word rather than in context, so a word
  can be flagged or missed depending on how it reads in isolation.
- The classifier reports a single offensiveness score per word/sentence.
  Splitting that into categories like profanity vs. insult vs. hate speech
  would need a model actually trained for multi-label toxicity (e.g.
  Detoxify's `unbiased` checkpoint), not just rescaling one score.
- Runs on CPU; expect noticeably longer processing time for videos beyond
  a minute or two.

## Live demo

_(link goes here once deployed)_