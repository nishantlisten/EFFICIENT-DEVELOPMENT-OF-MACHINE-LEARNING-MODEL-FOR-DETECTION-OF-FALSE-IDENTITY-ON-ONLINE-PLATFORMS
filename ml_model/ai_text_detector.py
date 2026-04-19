"""
=============================================================
  AI-Generated Text Detection (Heuristic Approach)
=============================================================
Analyzes text to determine likelihood of being AI-generated
using linguistic heuristics — no heavy dependencies required.

Heuristics used:
  1. Vocabulary diversity (Type-Token Ratio)
  2. Sentence length uniformity
  3. Repetitive phrase patterns
  4. Formality / generic phrasing indicators
  5. Excessive punctuation regularity
  6. Common AI filler phrases
"""

import re
import math
from collections import Counter


# ── Common AI-Generated Text Indicators ──────────────────────
AI_PHRASES = [
    "in today's world", "it is worth noting", "in conclusion",
    "furthermore", "moreover", "additionally", "it is important to",
    "in this article", "as we all know", "without a doubt",
    "it goes without saying", "needless to say", "at the end of the day",
    "in the realm of", "leveraging", "utilizing", "facilitate",
    "comprehensive", "robust", "innovative", "cutting-edge",
    "game-changer", "groundbreaking", "revolutionize", "delve",
    "tapestry", "beacon", "landscape", "paradigm", "synergy",
    "holistic", "nuanced", "multifaceted", "pivotal", "crucial",
    "foster", "harness", "navigate", "underscore", "testament",
    "embark", "unravel", "realm", "plethora", "myriad",
    "in essence", "it's important to note", "one must consider",
    "this is particularly", "it should be noted",
]

GENERIC_BIO_PATTERNS = [
    r"(?:passionate|dedicated|driven)\s+(?:about|to|by)",
    r"(?:lover|enthusiast|advocate)\s+of",
    r"(?:aspiring|budding|emerging)\s+\w+",
    r"living\s+(?:my|the)\s+(?:best|dream)",
    r"on\s+a\s+(?:journey|mission|quest)",
    r"making\s+the\s+world\s+a\s+better\s+place",
    r"here\s+to\s+(?:inspire|connect|share|help)",
]


def _tokenize(text: str) -> list[str]:
    """Simple whitespace + punctuation tokenizer."""
    return re.findall(r"\b\w+\b", text.lower())


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences."""
    sentences = re.split(r"[.!?]+", text)
    return [s.strip() for s in sentences if s.strip()]


def _vocabulary_diversity(tokens: list[str]) -> float:
    """
    Type-Token Ratio: unique words / total words.
    AI text tends to have lower diversity (more repetitive vocabulary).
    """
    if not tokens:
        return 1.0
    return len(set(tokens)) / len(tokens)


def _sentence_length_uniformity(sentences: list[str]) -> float:
    """
    Coefficient of variation of sentence lengths.
    AI text tends to have very uniform sentence lengths.
    Returns 0-1 where 1 = perfectly uniform (suspicious).
    """
    if len(sentences) < 2:
        return 0.0
    lengths = [len(s.split()) for s in sentences]
    mean = sum(lengths) / len(lengths)
    if mean == 0:
        return 0.0
    variance = sum((l - mean) ** 2 for l in lengths) / len(lengths)
    std = math.sqrt(variance)
    cv = std / mean
    # Low CV = uniform = suspicious → invert and cap
    uniformity = max(0, 1 - cv)
    return uniformity


def _ai_phrase_density(text: str) -> float:
    """
    Count AI-typical phrases per 100 words.
    """
    text_lower = text.lower()
    word_count = len(text_lower.split())
    if word_count == 0:
        return 0.0
    matches = sum(1 for phrase in AI_PHRASES if phrase in text_lower)
    return min(1.0, matches / max(word_count / 20, 1))


def _generic_pattern_score(text: str) -> float:
    """
    Check for generic bio/post patterns commonly seen in AI output.
    """
    text_lower = text.lower()
    matches = sum(
        1 for pattern in GENERIC_BIO_PATTERNS
        if re.search(pattern, text_lower)
    )
    return min(1.0, matches / 3)


def _punctuation_regularity(text: str) -> float:
    """
    Check if punctuation usage is suspiciously regular.
    AI text often has very consistent comma/period placement.
    """
    sentences = _split_sentences(text)
    if len(sentences) < 2:
        return 0.0
    # Check if all sentences end with similar punctuation pattern
    comma_counts = [s.count(",") for s in sentences]
    if not comma_counts:
        return 0.0
    mean_commas = sum(comma_counts) / len(comma_counts)
    if mean_commas == 0:
        return 0.0
    variance = sum((c - mean_commas) ** 2 for c in comma_counts) / len(comma_counts)
    cv = math.sqrt(variance) / mean_commas if mean_commas > 0 else 0
    return max(0, 1 - cv) * 0.5


def _repetitive_structure(sentences: list[str]) -> float:
    """
    Check for repetitive sentence openings (AI tends to start
    sentences with similar patterns).
    """
    if len(sentences) < 3:
        return 0.0
    # Get first 2 words of each sentence
    openers = []
    for s in sentences:
        words = s.split()[:2]
        if words:
            openers.append(" ".join(words).lower())
    if not openers:
        return 0.0
    counts = Counter(openers)
    most_common_count = counts.most_common(1)[0][1]
    repetition_ratio = most_common_count / len(openers)
    return min(1.0, repetition_ratio)


def detect_ai_text(text: str) -> dict:
    """
    Analyze text and return AI-generation probability.

    Args:
        text: The text to analyze (bio, post, article, etc.)

    Returns:
        dict with keys:
            - is_ai_generated: bool
            - confidence: float (0.0 to 1.0)
            - reasons: list of str explaining the verdict
            - scores: dict of individual heuristic scores
    """
    if not text or len(text.strip()) < 10:
        return {
            "is_ai_generated": False,
            "confidence": 0.0,
            "reasons": ["Text too short to analyze reliably."],
            "scores": {},
        }

    tokens = _tokenize(text)
    sentences = _split_sentences(text)

    # ── Calculate individual scores ──────────────────────────
    vocab_div = _vocabulary_diversity(tokens)
    sent_uniform = _sentence_length_uniformity(sentences)
    ai_phrase = _ai_phrase_density(text)
    generic = _generic_pattern_score(text)
    punct_reg = _punctuation_regularity(text)
    repetitive = _repetitive_structure(sentences)

    scores = {
        "vocabulary_diversity": round(vocab_div, 3),
        "sentence_uniformity": round(sent_uniform, 3),
        "ai_phrase_density": round(ai_phrase, 3),
        "generic_patterns": round(generic, 3),
        "punctuation_regularity": round(punct_reg, 3),
        "repetitive_structure": round(repetitive, 3),
    }

    # ── Weighted combination ─────────────────────────────────
    # Lower vocab diversity → more AI-like
    vocab_score = max(0, 1 - vocab_div) if len(tokens) > 20 else 0

    weights = {
        "vocab": 0.15,
        "uniformity": 0.20,
        "ai_phrases": 0.25,
        "generic": 0.15,
        "punctuation": 0.10,
        "repetitive": 0.15,
    }

    weighted_score = (
        weights["vocab"] * vocab_score
        + weights["uniformity"] * sent_uniform
        + weights["ai_phrases"] * ai_phrase
        + weights["generic"] * generic
        + weights["punctuation"] * punct_reg
        + weights["repetitive"] * repetitive
    )

    # Normalize to 0-1
    confidence = min(1.0, max(0.0, weighted_score))

    # ── Build explanations ───────────────────────────────────
    reasons = []

    if ai_phrase > 0.3:
        reasons.append("Contains multiple AI-typical phrases and buzzwords")
    if sent_uniform > 0.7:
        reasons.append("Suspiciously uniform sentence lengths")
    if vocab_score > 0.3:
        reasons.append("Low vocabulary diversity — repetitive word usage")
    if generic > 0.3:
        reasons.append("Uses generic/formulaic patterns common in AI output")
    if repetitive > 0.4:
        reasons.append("Repetitive sentence structure/openings detected")
    if punct_reg > 0.3:
        reasons.append("Unusually regular punctuation patterns")

    if not reasons:
        if confidence > 0.3:
            reasons.append("Mild indicators of AI-generated content detected")
        else:
            reasons.append("Text appears to be human-written")

    is_ai = confidence >= 0.4

    return {
        "is_ai_generated": is_ai,
        "confidence": round(confidence, 3),
        "reasons": reasons,
        "scores": scores,
    }


# ── Quick test ────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  AI TEXT DETECTION – DEMO")
    print("=" * 60)

    # Test with AI-like text
    ai_text = (
        "In today's world, it is important to leverage innovative "
        "strategies to navigate the complex landscape of social media. "
        "Furthermore, one must consider the multifaceted nature of "
        "digital communication. Moreover, this comprehensive approach "
        "fosters holistic engagement and underscores the pivotal role "
        "of robust content creation in our ever-evolving digital realm."
    )

    # Test with human-like text
    human_text = (
        "hey guys!! just got back from the beach lol 🏖️ "
        "best day ever tbh. pizza was amazing and jake fell "
        "off his surfboard AGAIN 😂😂 gonna post more pics later"
    )

    # Test with a bio
    ai_bio = (
        "Passionate about leveraging technology to make the world "
        "a better place. Dedicated to fostering innovation and "
        "driving meaningful change. Here to inspire and connect."
    )

    tests = [
        ("AI-like paragraph", ai_text),
        ("Human-like post", human_text),
        ("AI-like bio", ai_bio),
    ]

    for name, text in tests:
        result = detect_ai_text(text)
        status = "🤖 AI" if result["is_ai_generated"] else "👤 Human"
        print(f"\n{'─' * 50}")
        print(f"  Test: {name}")
        print(f"  Text: {text[:80]}...")
        print(f"  Verdict: {status}  (confidence: {result['confidence']:.1%})")
        print(f"  Reasons:")
        for r in result["reasons"]:
            print(f"    • {r}")

    print(f"\n{'=' * 60}")
    print("  ✅ AI TEXT DETECTION TESTS COMPLETE")
    print("=" * 60)
