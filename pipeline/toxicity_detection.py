"""Toxicity / offensive-language classification using a pretrained transformer.

Changes from the original notebook:

1. The model and tokenizer were loaded twice (once for word-level
   classification, once for sentence-level). They're now loaded once and
   cached.
2. Words (and sentences) were scored one at a time -- a full tokenize +
   forward pass per word. On CPU, per-call overhead dominates, so a
   transcript with a hundred-plus words meant a hundred-plus separate
   passes. Scoring now happens in batches, which cuts that down to a
   handful of passes.
3. The notebook derived "insult" and "hate_speech" scores by multiplying
   the single offensive-language score by 0.8 and 0.6 -- those aren't real
   separate classifications, just the same number rescaled, which would be
   misleading to present as multi-label output. This reports one honest
   offensiveness score instead. A genuinely multi-label model (e.g.
   Detoxify's `unbiased` checkpoint, which scores toxicity, severe
   toxicity, obscenity, threat, insult, and identity attack separately) is
   the right upgrade if that breakdown matters later.
"""
import re
from functools import lru_cache
from typing import Dict, List

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_NAME = "cardiffnlp/twitter-roberta-base-offensive"
OFFENSIVE_THRESHOLD = 0.4
WORD_BATCH_SIZE = 64
SENTENCE_BATCH_SIZE = 16


@lru_cache(maxsize=1)
def _load_classifier():
    """Load and cache the tokenizer + model so it's only loaded once per process."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    model.eval()
    return tokenizer, model


def _score_batch(texts: List[str]) -> List[float]:
    """Return the offensive-class probability for each text, in one forward pass."""
    tokenizer, model = _load_classifier()
    inputs = tokenizer(texts, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    return torch.softmax(outputs.logits, dim=1)[:, 1].tolist()


def _normalize(word: str) -> str:
    """Lowercase a word and strip surrounding punctuation.

    Whisper's word-level output keeps punctuation attached (e.g. "word!"),
    so normalizing both sides the same way is what lets toxic words be
    matched back to their timestamps later.
    """
    return re.sub(r"^\W+|\W+$", "", word).lower()


def identify_toxic_words(
    text: str, threshold: float = OFFENSIVE_THRESHOLD, batch_size: int = WORD_BATCH_SIZE
) -> List[str]:
    """Return the normalized set of words in ``text`` classified as offensive.

    Each word is still scored independently (matching the original
    notebook's per-word approach), just in batches rather than one at a time.
    """
    words = text.split()
    if not words:
        return []

    toxic = set()
    for i in range(0, len(words), batch_size):
        chunk = words[i : i + batch_size]
        scores = _score_batch(chunk)
        for word, score in zip(chunk, scores):
            if score > threshold:
                toxic.add(_normalize(word))
    return sorted(toxic)


def sentence_report(sentences: List[str], batch_size: int = SENTENCE_BATCH_SIZE) -> Dict:
    """Score each sentence for offensiveness and summarize.

    Returns:
        A dict with "max_score", "average_score", and "per_sentence"
        (a list of (sentence, score) pairs).
    """
    if not sentences:
        return {"max_score": 0.0, "average_score": 0.0, "per_sentence": []}

    scored = []
    for i in range(0, len(sentences), batch_size):
        chunk = sentences[i : i + batch_size]
        scores = _score_batch(chunk)
        scored.extend(zip(chunk, scores))

    values = [score for _, score in scored]
    return {
        "max_score": max(values),
        "average_score": sum(values) / len(values),
        "per_sentence": scored,
    }