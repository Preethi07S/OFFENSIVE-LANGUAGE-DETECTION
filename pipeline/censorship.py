"""Censor toxic words in audio with a beep, then merge the result back into
the source video.

The original notebook stopped at a censored .wav file the user had to
download separately. ``merge_audio_into_video`` closes that gap so the
pipeline's real output is a finished video, not a loose audio file.
"""
from typing import Dict, List

from moviepy.editor import AudioFileClip, VideoFileClip
from pydub import AudioSegment
from pydub.generators import Sine


def generate_beep(duration_ms: int = 500, freq: int = 1000) -> AudioSegment:
    """Generate a sine-wave beep tone of the given duration and frequency."""
    return Sine(freq).to_audio_segment(duration=duration_ms)


def match_toxic_timestamps(words_with_timestamps: List[Dict], toxic_words: List[str]) -> List[Dict]:
    """Find the timestamp entries whose normalized word is in ``toxic_words``.

    Args:
        words_with_timestamps: Output of transcription.transcribe()["words"].
        toxic_words: Normalized toxic words from toxicity_detection.identify_toxic_words().

    Returns:
        The subset of ``words_with_timestamps`` that matched, in original order.
    """
    toxic_set = set(toxic_words)
    matches = []
    for word_info in words_with_timestamps:
        normalized = word_info["word"].strip().lower().strip(".,!?\"'")
        if normalized in toxic_set:
            matches.append(word_info)
    return matches


def censor_audio(audio_path: str, toxic_timestamps: List[Dict], output_path: str) -> str:
    """Replace each toxic word's audio span with a beep tone and export the result."""
    audio = AudioSegment.from_wav(audio_path)

    for word_info in sorted(toxic_timestamps, key=lambda w: w["start_ms"]):
        start_ms = int(word_info["start_ms"])
        end_ms = int(word_info["end_ms"])
        beep = generate_beep(duration_ms=end_ms - start_ms)
        audio = audio[:start_ms] + beep + audio[end_ms:]

    audio.export(output_path, format="wav")
    return output_path


def merge_audio_into_video(video_path: str, censored_audio_path: str, output_path: str) -> str:
    """Replace a video's audio track with the censored audio and export the merged video.

    This is the step the original notebook was missing -- without it, a
    viewer has a censored audio file but no way to watch the actual video
    with the censorship applied.
    """
    video = VideoFileClip(video_path)
    new_audio = AudioFileClip(censored_audio_path)
    try:
        video = video.set_audio(new_audio)
        video.write_videofile(output_path, logger=None, audio_codec="aac")
    finally:
        video.close()
        new_audio.close()
    return output_path
