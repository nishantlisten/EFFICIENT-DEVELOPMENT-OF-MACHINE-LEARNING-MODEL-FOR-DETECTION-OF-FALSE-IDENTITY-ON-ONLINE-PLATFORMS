"""
=============================================================
  Multi-Signal Behavior & Content Consistency Module
=============================================================
Analyzes a list of posts (text, image, video) + profile data
to detect fake/bot/scripted behavior patterns.

Signals Produced:
  - posts_per_day            (temporal behavior)
  - time_variance            (posting regularity — bots are too regular)
  - engagement_ratio         (likes / followers)
  - duplicate_text_ratio     (TF-IDF cosine similarity across captions)
  - ai_text_score            (AI/scripted language patterns in posts)
  - agenda_score             (single-topic keyword/hashtag push)
  - content_similarity_score (bio vs posts topic alignment)
  - face_presence_ratio      (how many post images contain a face)

Video/Audio:
  - FFmpeg extracts audio from video posts (if available)
  - Whisper (tiny model, local) transcribes audio → text
  - Transcribed text is appended to captions for NLP analysis

All signals are lightweight — no heavy deep learning required.
=============================================================
"""

import re
import os
import sys
import math
import tempfile
import subprocess
import statistics
from io import BytesIO
from datetime import datetime
from collections import Counter

import numpy as np
import requests

# ── Optional heavy imports (gracefully degrade) ───────────────
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    _SKLEARN_OK = True
except ImportError:
    _SKLEARN_OK = False

try:
    import whisper as _whisper
    _WHISPER_MODEL = None  # loaded lazily on first use
    _WHISPER_OK = True
except ImportError:
    _WHISPER_OK = False
    _WHISPER_MODEL = None

try:
    from PIL import Image
    import cv2
    import imagehash
    _CV2_OK = True
except ImportError:
    _CV2_OK = False

# Re-use the existing image_module if available
_IMAGE_MODULE_OK = False
try:
    _proj_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _proj_root not in sys.path:
        sys.path.insert(0, _proj_root)
    from ml_model.image_module import analyze_image
    _IMAGE_MODULE_OK = True
except Exception:
    analyze_image = None

# ── Constants ─────────────────────────────────────────────────
SPAM_KEYWORDS = [
    "earn money", "link in bio", "dm me", "free", "giveaway",
    "click link", "make money", "work from home", "limited offer",
    "discount", "promo code", "shop now", "buy now",
    "follow back", "followback", "f4f", "l4l",
    "subscribe", "check bio", "check link",
]

AGENDA_KEYWORDS = [
    "vote", "election", "truth", "expose", "wake up",
    "conspiracy", "propaganda", "deep state", "join us",
    "revolution", "movement", "share this",
]

AI_POST_PHRASES = [
    "in today's world", "leveraging", "groundbreaking",
    "innovative solutions", "holistic approach", "synergy",
    "delve", "tapestry", "comprehensive", "pivotal",
    "foster", "underscore", "navigate", "harness",
    "multifaceted", "nuanced", "robust", "cutting-edge",
]


# ═══════════════════════════════════════════════════════════════
#  SECTION 1 — VIDEO / AUDIO TRANSCRIPTION
# ═══════════════════════════════════════════════════════════════

def _get_whisper_model():
    """Lazy-load Whisper tiny model once."""
    global _WHISPER_MODEL
    if _WHISPER_MODEL is None and _WHISPER_OK:
        try:
            import whisper
            _WHISPER_MODEL = whisper.load_model("tiny")
        except Exception as e:
            print(f"[BEHAVIOR] Whisper load error: {e}")
    return _WHISPER_MODEL


def _check_ffmpeg() -> bool:
    """Check if FFmpeg is available on the system PATH."""
    try:
        result = subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True, timeout=5
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def transcribe_video_audio(video_url: str) -> str:
    """
    Download a video, extract audio via FFmpeg, transcribe with Whisper.
    Returns transcribed text string, or empty string on any failure.
    Gracefully skips if FFmpeg or Whisper are unavailable.
    """
    if not _WHISPER_OK or not _check_ffmpeg():
        return ""
    if not video_url:
        return ""

    model = _get_whisper_model()
    if model is None:
        return ""

    tmp_video = None
    tmp_audio = None
    try:
        # Download video to temp file
        headers = {"User-Agent": "Mozilla/5.0"}
        resp = requests.get(video_url, headers=headers, timeout=15, stream=True)
        if resp.status_code != 200:
            return ""

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as vf:
            for chunk in resp.iter_content(chunk_size=8192):
                vf.write(chunk)
            tmp_video = vf.name

        # Extract audio with FFmpeg
        tmp_audio = tmp_video.replace(".mp4", ".wav")
        cmd = [
            "ffmpeg", "-y", "-i", tmp_video,
            "-ar", "16000", "-ac", "1",
            "-c:a", "pcm_s16le", tmp_audio,
            "-loglevel", "error"
        ]
        subprocess.run(cmd, timeout=60, check=True)

        # Transcribe
        import whisper
        result = model.transcribe(tmp_audio, fp16=False)
        return result.get("text", "").strip()

    except Exception as e:
        print(f"[BEHAVIOR] Video transcription error for {video_url[:60]}: {e}")
        return ""
    finally:
        for f in [tmp_video, tmp_audio]:
            if f and os.path.exists(f):
                try:
                    os.remove(f)
                except Exception:
                    pass


# ═══════════════════════════════════════════════════════════════
#  SECTION 2 — TEXT & AI ANALYSIS
# ═══════════════════════════════════════════════════════════════

def _tokenize(text: str) -> list:
    return re.findall(r"\b\w+\b", text.lower())


def _lexical_diversity(text: str) -> float:
    """Type-Token Ratio. Higher = more diverse vocabulary."""
    tokens = _tokenize(text)
    if len(tokens) < 5:
        return 1.0
    return len(set(tokens)) / len(tokens)


def _ai_phrase_score(texts: list) -> float:
    """Fraction of AI-typical phrases found across all post texts."""
    combined = " ".join(texts).lower()
    if not combined.strip():
        return 0.0
    word_count = len(combined.split())
    hits = sum(1 for p in AI_POST_PHRASES if p in combined)
    return min(1.0, hits / max(word_count / 30, 1))


def _duplicate_text_ratio(texts: list) -> float:
    """
    Average pairwise TF-IDF cosine similarity between post captions.
    High value = repetitive/scripted content.
    Falls back to simple exact-match ratio if sklearn missing.
    """
    clean = [t.strip() for t in texts if t and len(t.strip()) > 5]
    if len(clean) < 2:
        return 0.0

    if _SKLEARN_OK:
        try:
            vec = TfidfVectorizer(min_df=1, stop_words="english")
            matrix = vec.fit_transform(clean)
            sim = cosine_similarity(matrix)
            # Upper triangle (excluding diagonal)
            n = len(clean)
            pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
            if not pairs:
                return 0.0
            avg_sim = sum(sim[i][j] for i, j in pairs) / len(pairs)
            return round(float(avg_sim), 3)
        except Exception:
            pass

    # Fallback: exact duplicate ratio
    from collections import Counter
    counts = Counter(clean)
    duplicates = sum(c - 1 for c in counts.values())
    return min(1.0, duplicates / len(clean))


def _sentence_length_uniformity(text: str) -> float:
    """
    Coefficient of variation of sentence lengths — low CV = suspiciously uniform.
    Returns 0–1 where 1 = perfectly uniform.
    """
    sentences = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]
    if len(sentences) < 2:
        return 0.0
    lengths = [len(s.split()) for s in sentences]
    mean = sum(lengths) / len(lengths)
    if mean == 0:
        return 0.0
    variance = sum((l - mean) ** 2 for l in lengths) / len(lengths)
    std = math.sqrt(variance)
    cv = std / mean
    return round(max(0.0, 1.0 - cv), 3)


# ═══════════════════════════════════════════════════════════════
#  SECTION 3 — AGENDA DETECTION
# ═══════════════════════════════════════════════════════════════

def _extract_hashtags(text: str) -> list:
    return re.findall(r"#\w+", text.lower())


def _agenda_score(texts: list) -> float:
    """
    Measures how strongly the account pushes a single agenda:
    - Spam keyword density
    - Agenda keyword density
    - Hashtag repetition (few unique hashtags used repeatedly)
    Returns 0–1.
    """
    if not texts:
        return 0.0

    combined = " ".join(texts).lower()
    word_count = max(len(combined.split()), 1)

    # Spam keyword hits
    spam_hits = sum(1 for kw in SPAM_KEYWORDS if kw in combined)
    spam_score = min(1.0, spam_hits / 3)

    # Agenda keyword hits
    agenda_hits = sum(1 for kw in AGENDA_KEYWORDS if kw in combined)
    agenda_kw_score = min(1.0, agenda_hits / 3)

    # Hashtag repetition: ratio of unique to total hashtags
    all_tags = []
    for t in texts:
        all_tags.extend(_extract_hashtags(t))
    if len(all_tags) > 1:
        unique_ratio = len(set(all_tags)) / len(all_tags)
        hashtag_rep = max(0.0, 1.0 - unique_ratio)  # high = repetitive
    else:
        hashtag_rep = 0.0

    # Topic repetition via top keyword concentration
    tokens = _tokenize(combined)
    stopwords = {
        "the", "is", "in", "it", "of", "and", "a", "to", "was",
        "for", "on", "are", "with", "as", "at", "be", "by",
        "this", "that", "have", "from", "or", "an", "but",
        "not", "you", "my", "we", "i", "me", "he", "she",
    }
    content_tokens = [t for t in tokens if t not in stopwords and len(t) > 3]
    if content_tokens:
        counts = Counter(content_tokens)
        top5 = sum(c for _, c in counts.most_common(5))
        topic_concentration = top5 / len(content_tokens)
        topic_score = min(1.0, topic_concentration * 2)
    else:
        topic_score = 0.0

    final = (spam_score * 0.35 + agenda_kw_score * 0.25 +
             hashtag_rep * 0.20 + topic_score * 0.20)
    return round(final, 3)


# ═══════════════════════════════════════════════════════════════
#  SECTION 4 — IMAGE / FACE CONSISTENCY
# ═══════════════════════════════════════════════════════════════

def _compute_face_presence_ratio(posts: list) -> tuple:
    """
    For each image/video post with a media_url, run analyze_image().
    Returns (face_presence_ratio, image_reuse_detected).
    """
    if not _IMAGE_MODULE_OK or not analyze_image:
        return 0.0, False

    image_posts = [
        p for p in posts
        if p.get("media_type", "image") in ("image", "photo")
        and p.get("media_url", "")
    ]
    if not image_posts:
        return 0.0, False

    face_count = 0
    reuse_detected = False

    for post in image_posts[:10]:  # Limit to 10 to avoid slow API
        try:
            result = analyze_image(post["media_url"])
            if result.get("has_face"):
                face_count += 1
            if result.get("same_image_reuse"):
                reuse_detected = True
        except Exception:
            continue

    ratio = face_count / len(image_posts) if image_posts else 0.0
    return round(ratio, 3), reuse_detected


# ═══════════════════════════════════════════════════════════════
#  SECTION 5 — CROSS-CONTENT SIMILARITY
# ═══════════════════════════════════════════════════════════════

def _cosine_sim_texts(text_a: str, text_b: str) -> float:
    """TF-IDF cosine similarity between two text blocks."""
    if not _SKLEARN_OK:
        return 0.0
    a, b = text_a.strip(), text_b.strip()
    if not a or not b:
        return 0.0
    try:
        vec = TfidfVectorizer(min_df=1, stop_words="english")
        matrix = vec.fit_transform([a, b])
        sim = cosine_similarity(matrix[0:1], matrix[1:2])
        return round(float(sim[0][0]), 3)
    except Exception:
        return 0.0


def _content_similarity_score(bio: str, captions: list) -> float:
    """
    Measures similarity between bio topic and post topics.
    Genuine profiles usually have a topic coherence between bio & posts.
    Very HIGH similarity (> 0.9) = suspiciously uniform (scripted).
    Very LOW (< 0.05) = bio and posts seem completely unrelated (fake).
    Returns raw similarity score 0–1.
    """
    if not captions:
        return 0.0
    combined_posts = " ".join(captions)
    return _cosine_sim_texts(bio, combined_posts)


# ═══════════════════════════════════════════════════════════════
#  SECTION 6 — BEHAVIOR & TEMPORAL ANALYSIS
# ═══════════════════════════════════════════════════════════════

def _parse_timestamp(ts) -> datetime | None:
    """Parse ISO str, Unix int, or datetime. Returns None on failure."""
    if ts is None:
        return None
    if isinstance(ts, datetime):
        return ts
    if isinstance(ts, (int, float)):
        try:
            return datetime.utcfromtimestamp(ts)
        except Exception:
            return None
    if isinstance(ts, str):
        for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S",
                    "%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%d"):
            try:
                return datetime.strptime(ts, fmt)
            except ValueError:
                continue
    return None


def _temporal_behavior(posts: list) -> dict:
    """
    Compute:
    - posts_per_day: average daily posting rate
    - avg_time_gap_hours: mean gap between consecutive posts (hours)
    - posting_time_variance: std deviation of posting hour (0–23)
    - time_variance_score: 0–1 normalized regularity score
    """
    timestamps = []
    for p in posts:
        ts = _parse_timestamp(p.get("timestamp"))
        if ts:
            timestamps.append(ts)

    if len(timestamps) < 2:
        return {
            "posts_per_day": 0.0,
            "avg_time_gap_hours": 0.0,
            "posting_time_variance": 12.0,  # max uncertainty
            "time_variance_score": 0.0,
        }

    timestamps.sort()
    span_days = max(
        (timestamps[-1] - timestamps[0]).total_seconds() / 86400, 1
    )
    posts_per_day = round(len(timestamps) / span_days, 3)

    gaps = [
        (timestamps[i] - timestamps[i - 1]).total_seconds() / 3600
        for i in range(1, len(timestamps))
    ]
    avg_gap = round(statistics.mean(gaps), 3)

    hours = [dt.hour for dt in timestamps]
    if len(hours) >= 2:
        hour_std = round(statistics.stdev(hours), 3)
    else:
        hour_std = 12.0

    # Low std → highly regular posting times → bot-like
    # Normalise: std of 0 = score 1.0, std of 12 = score 0.0
    time_variance_score = round(max(0.0, 1.0 - (hour_std / 12.0)), 3)

    return {
        "posts_per_day": posts_per_day,
        "avg_time_gap_hours": avg_gap,
        "posting_time_variance": hour_std,
        "time_variance_score": time_variance_score,
    }


def _engagement_ratio(posts: list, num_followers: int) -> float:
    """Average (likes / followers) across posts. Low = inauthentic."""
    if num_followers <= 0 or not posts:
        return 0.0
    likes = [float(p.get("likes", 0)) for p in posts]
    avg_likes = sum(likes) / len(likes)
    return round(avg_likes / num_followers, 4)


# ═══════════════════════════════════════════════════════════════
#  SECTION 7 — MASTER ANALYSIS FUNCTION
# ═══════════════════════════════════════════════════════════════

def analyze_behavior(profile: dict, posts: list) -> dict:
    """
    Run full multi-signal behavior analysis.

    Args:
        profile: dict with keys:
            - bio (str)
            - num_followers (int)
            - profile_pic_url (str, optional)
        posts: list of dicts, each with keys:
            - timestamp (str/int/datetime)
            - caption (str)
            - likes (int)
            - media_type ("image" | "video" | "photo")
            - media_url (str, optional)

    Returns:
        dict with all signals + reasons list
    """
    bio = profile.get("bio", "") or ""
    num_followers = int(profile.get("num_followers", 0))
    reasons = []
    video_transcripts = []

    # ── Step 1: Transcribe video audio ───────────────────────
    ffmpeg_available = _check_ffmpeg()
    whisper_available = _WHISPER_OK
    for post in posts:
        if post.get("media_type") == "video" and post.get("media_url"):
            transcript = transcribe_video_audio(post["media_url"])
            if transcript:
                video_transcripts.append(transcript)
                # Augment caption with spoken text
                existing = post.get("caption", "") or ""
                post["_augmented_caption"] = (
                    existing + " " + transcript if existing else transcript
                )

    # ── Step 2: Collect all caption texts ────────────────────
    captions = []
    for post in posts:
        text = post.get("_augmented_caption") or post.get("caption") or ""
        text = text.strip()
        if text:
            captions.append(text)

    all_text = " ".join(captions)

    # ── Step 3: Text / AI scores ──────────────────────────────
    ai_text_score = _ai_phrase_score(captions)
    dup_ratio = _duplicate_text_ratio(captions)
    lex_div = _lexical_diversity(all_text) if all_text else 1.0
    sent_uniformity = _sentence_length_uniformity(all_text) if all_text else 0.0

    # Composite AI text score (weighted)
    composite_ai = round(
        ai_text_score * 0.40 +
        (1.0 - lex_div) * 0.30 +
        sent_uniformity * 0.30,
        3
    )

    # ── Step 4: Agenda score ──────────────────────────────────
    agenda = _agenda_score(captions + ([bio] if bio else []))

    # ── Step 5: Image / face consistency ──────────────────────
    face_ratio, image_reuse = _compute_face_presence_ratio(posts)

    # ── Step 6: Cross-content similarity ──────────────────────
    content_sim = _content_similarity_score(bio, captions)

    # ── Step 7: Temporal behavior ────────────────────────────
    temporal = _temporal_behavior(posts)

    # ── Step 8: Engagement ratio ─────────────────────────────
    eng_ratio = _engagement_ratio(posts, num_followers)

    # ── Step 9: Build explanation reasons ────────────────────
    ppd = temporal["posts_per_day"]
    tvar = temporal["time_variance_score"]

    if ppd > 10:
        reasons.append(
            f"🤖 Extremely high posting frequency ({ppd:.1f} posts/day) — "
            "likely automated"
        )
    elif ppd > 5:
        reasons.append(
            f"⚠️ High posting frequency ({ppd:.1f} posts/day)"
        )

    if tvar > 0.75 and ppd > 2:
        reasons.append(
            "🤖 Posts at near-identical times — strongly suggests scheduled bot"
        )
    elif tvar > 0.55:
        reasons.append("⚠️ Suspiciously regular posting schedule")

    if dup_ratio > 0.70:
        reasons.append(
            f"🤖 Highly repetitive captions (similarity {dup_ratio:.0%}) — "
            "scripted/copy-paste content"
        )
    elif dup_ratio > 0.45:
        reasons.append(
            f"⚠️ Moderately repetitive post captions ({dup_ratio:.0%} similarity)"
        )

    if composite_ai > 0.55:
        reasons.append(
            "🤖 Post content shows strong AI-generated language patterns"
        )
    elif composite_ai > 0.30:
        reasons.append(
            "⚠️ Some AI-like phrasing detected in post captions"
        )

    if agenda > 0.60:
        reasons.append(
            "🚨 Content shows strong single-topic agenda (spam/promotion/political)"
        )
    elif agenda > 0.35:
        reasons.append("⚠️ Noticeable topic agenda or spam keyword density")

    if image_reuse:
        reasons.append(
            "🚨 Post images appear to be reused/duplicate photos"
        )

    if len(posts) >= 3 and face_ratio < 0.20:
        reasons.append(
            f"⚠️ Only {face_ratio:.0%} of post images contain a human face"
        )

    if eng_ratio < 0.005 and num_followers > 500:
        reasons.append(
            f"🤖 Extremely low engagement ratio ({eng_ratio:.4f}) "
            "despite high follower count — suggests fake followers"
        )
    elif eng_ratio < 0.01 and num_followers > 200:
        reasons.append(
            f"⚠️ Low engagement ratio ({eng_ratio:.4f}) — "
            "followers may not be genuine"
        )

    if content_sim > 0.92 and len(captions) >= 3:
        reasons.append(
            "🤖 Suspiciously uniform content across bio and all posts"
        )
    elif content_sim < 0.03 and bio and len(captions) >= 2:
        reasons.append(
            "⚠️ Bio topic is completely unrelated to post content"
        )

    if video_transcripts:
        reasons.append(
            f"🎙️ Audio transcribed from {len(video_transcripts)} video post(s) "
            "and included in analysis"
        )

    if not ffmpeg_available:
        reasons.append(
            "ℹ️ FFmpeg not found — video audio transcription skipped"
        )
    if not whisper_available:
        reasons.append(
            "ℹ️ Whisper not installed — video audio transcription unavailable"
        )
    if not _SKLEARN_OK:
        reasons.append(
            "ℹ️ scikit-learn not available — advanced similarity analysis skipped"
        )

    return {
        # Core output signals
        "posts_per_day": ppd,
        "time_variance_score": tvar,
        "avg_time_gap_hours": temporal["avg_time_gap_hours"],
        "posting_time_variance": temporal["posting_time_variance"],
        "engagement_ratio": eng_ratio,
        "duplicate_text_ratio": dup_ratio,
        "lexical_diversity": round(lex_div, 3),
        "sentence_uniformity": sent_uniformity,
        "ai_text_score": composite_ai,
        "agenda_score": agenda,
        "content_similarity_score": content_sim,
        "face_presence_ratio": face_ratio,
        "image_reuse_detected": image_reuse,
        "video_transcripts_count": len(video_transcripts),
        "ffmpeg_available": ffmpeg_available,
        "whisper_available": whisper_available,
        # Explanation
        "reasons": reasons,
        "post_count_analyzed": len(posts),
    }


# ═══════════════════════════════════════════════════════════════
#  SECTION 8 — RULE-BASED PROBABILITY ADJUSTER
# ═══════════════════════════════════════════════════════════════

def apply_behavior_adjustments(
    behavior: dict, fake_prob: float, reasons: list
) -> tuple:
    """
    Apply rule-based fake probability adjustments based on behavior signals.

    Args:
        behavior: output dict from analyze_behavior()
        fake_prob: current fake probability (0.0–1.0) from ML model
        reasons: list of reason strings to append to

    Returns:
        (adjusted_fake_prob: float, reasons: list)
    """
    adj = fake_prob

    ppd = behavior.get("posts_per_day", 0)
    tvar = behavior.get("time_variance_score", 0)
    dup = behavior.get("duplicate_text_ratio", 0)
    ai_s = behavior.get("ai_text_score", 0)
    agenda = behavior.get("agenda_score", 0)
    content_sim = behavior.get("content_similarity_score", 0)
    eng = behavior.get("engagement_ratio", 1)
    face_ratio = behavior.get("face_presence_ratio", 1)
    image_reuse = behavior.get("image_reuse_detected", False)
    post_count = behavior.get("post_count_analyzed", 0)

    # Rule 1: High posting frequency + low time variance → bot pattern
    if ppd > 10 and tvar > 0.75:
        adj = min(0.99, adj + 0.20)
        reasons.append("🤖 Automated bot pattern: extreme frequency + fixed schedule")
    elif ppd > 5 and tvar > 0.55:
        adj = min(0.99, adj + 0.12)
        reasons.append("⚠️ High posting rate with suspiciously regular timing")

    # Rule 2: Duplicate content + AI score → scripted/spam
    if dup > 0.70 and ai_s > 0.50:
        adj = min(0.99, adj + 0.22)
        reasons.append("🤖 Repetitive AI-like content detected across posts")
    elif dup > 0.45 or ai_s > 0.45:
        adj = min(0.99, adj + 0.10)
        reasons.append("⚠️ Repetitive or AI-generated content patterns found")

    # Rule 3: Image reuse + low engagement → strong fake
    if image_reuse and eng < 0.01:
        adj = min(0.99, adj + 0.25)
        reasons.append("🚨 Reused post images combined with virtually zero engagement")
    elif image_reuse:
        adj = min(0.99, adj + 0.12)
        reasons.append("🚨 Duplicate/reused post images detected")

    # Rule 4: Strong single-topic agenda + uniform content
    if agenda > 0.60 and content_sim > 0.70:
        adj = min(0.99, adj + 0.15)
        reasons.append("🚨 Content shows strong single-topic agenda with scripted uniformity")
    elif agenda > 0.35:
        adj = min(0.99, adj + 0.06)
        reasons.append("⚠️ Agenda-driven content pattern (spam/promotional/political)")

    # Rule 5: Low face presence with many posts → stock images or no identity
    if post_count >= 5 and face_ratio < 0.15:
        adj = min(0.99, adj + 0.08)
        reasons.append("⚠️ Very few post images show a human face — possible stock imagery")

    # Rule 6: Extremely low engagement with many followers
    if eng < 0.003 and post_count >= 3:
        adj = min(0.99, adj + 0.10)
        reasons.append("🤖 Near-zero engagement ratio suggests bulk/purchased followers")

    return round(adj, 4), reasons
