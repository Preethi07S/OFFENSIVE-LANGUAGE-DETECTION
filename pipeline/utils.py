"""Small shared helpers used across pipeline stages."""
import re
from typing import List

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def split_sentences(text: str) -> List[str]:
    """Split text into sentences on '.', '!', or '?' followed by whitespace.

    This is a lightweight stand-in for the notebook's spacy-based sentence
    splitter. spacy (and the unused nltk 'punkt' download) added real
    deployment weight for a task a regex handles well enough here.
    """
    return [s.strip() for s in _SENTENCE_SPLIT_RE.split(text.strip()) if s.strip()]
