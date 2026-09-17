"""Speech-to-text transcription with word-level timestamps.

The original notebook called Whisper twice: once for plain text, once more
for timestamps. This does both in a single pass and caches the loaded model
so repeated calls (e.g. from a UI) don't reload it from disk each time.
"""
from functools import lru_cache
from typing import Dict, List

import whisper


@lru_cache(maxsize=1)
def _load_model(model_size: str = "base"):
    """Load and cache the Whisper model so it's only loaded once per process."""
    return whisper.load_model(model_size)


def transcribe(audio_path: str, model_size: str = "base") -> Dict:
    """Transcribe audio to text and word-level timestamps in a single pass.

    Args:
        audio_path: Path to the input audio file.
        model_size: Whisper model size (tiny/base/small/medium/large).
            "base" is a reasonable speed/accuracy tradeoff for a free-tier
            deployment; "small" trades some latency for better accuracy.

    Returns:
        A dict with:
            - "text": full transcript (str)
            - "words": list of {"word", "start_ms", "end_ms"} dicts
    """
    model = _load_model(model_size)
    result = model.transcribe(audio_path, word_timestamps=True)

    words: List[Dict] = []
    for segment in result["segments"]:
        for word_info in segment.get("words", []):
            words.append(
                {
                    "word": word_info["word"],
                    "start_ms": word_info["start"] * 1000,
                    "end_ms": word_info["end"] * 1000,
                }
            )

    return {"text": result["text"], "words": words}
