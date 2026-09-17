"""Offensive Language Detection pipeline.

Chains: video -> audio -> transcript + word timestamps -> toxicity
detection -> timestamp matching -> beep censorship -> merged output video.

Each stage is also importable on its own (see audio_extraction,
transcription, toxicity_detection, censorship) if you only need one step.
"""
import os

from .audio_extraction import extract_audio
from .transcription import transcribe
from .toxicity_detection import identify_toxic_words, sentence_report
from .censorship import match_toxic_timestamps, censor_audio, merge_audio_into_video
from .utils import split_sentences

__all__ = [
    "extract_audio",
    "transcribe",
    "identify_toxic_words",
    "sentence_report",
    "match_toxic_timestamps",
    "censor_audio",
    "merge_audio_into_video",
    "split_sentences",
    "run_pipeline",
]


def run_pipeline(video_path: str, work_dir: str = "workdir") -> dict:
    """Run the full pipeline on a video file.

    Args:
        video_path: Path to the input video.
        work_dir: Directory to write intermediate and output files to.
            Created if it doesn't exist.

    Returns:
        A dict with:
            - "transcript": full transcript text
            - "toxic_words": normalized toxic words detected
            - "toxic_timestamps": timestamp entries matched for those words
            - "censored_audio_path": path to the beeped-out audio file
            - "censored_video_path": path to the final video with censored audio
            - "report": sentence-level offensiveness scores
    """
    os.makedirs(work_dir, exist_ok=True)

    audio_path = extract_audio(video_path, os.path.join(work_dir, "audio.wav"))

    transcription = transcribe(audio_path)
    text, words = transcription["text"], transcription["words"]

    toxic_words = identify_toxic_words(text)
    report = sentence_report(split_sentences(text))

    toxic_timestamps = match_toxic_timestamps(words, toxic_words)
    censored_audio_path = censor_audio(
        audio_path, toxic_timestamps, os.path.join(work_dir, "censored_audio.wav")
    )
    censored_video_path = merge_audio_into_video(
        video_path, censored_audio_path, os.path.join(work_dir, "censored_video.mp4")
    )

    return {
        "transcript": text,
        "toxic_words": toxic_words,
        "toxic_timestamps": toxic_timestamps,
        "censored_audio_path": censored_audio_path,
        "censored_video_path": censored_video_path,
        "report": report,
    }
