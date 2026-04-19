"""
API Views for Fake Profile Detection System
=============================================
Endpoints:
  POST /api/predict-profile/  → Predict fake/real from profile features
  POST /api/detect-ai-text/   → Detect AI-generated text
  POST /api/analyze/           → Combined analysis
"""

import os
import sys
import numpy as np
import joblib
from django.conf import settings
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status

# ── Add project root to path so we can import ml_model ────────
PROJECT_ROOT = str(settings.PROJECT_ROOT)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ml_model.ai_text_detector import detect_ai_text
from ml_model.profile_scraper import scrape_instagram_profile
from ml_model.image_module import analyze_image
from ml_model.behavior_module import analyze_behavior, apply_behavior_adjustments

# ── Load ML model, scaler, and feature columns at startup ────
MODEL_DIR = settings.ML_MODEL_DIR

try:
    rf_model = joblib.load(os.path.join(MODEL_DIR, "fake_profile_model.pkl"))
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
    feature_columns = joblib.load(os.path.join(MODEL_DIR, "feature_columns.pkl"))
    print("[SUCCESS] ML model loaded successfully!")
except Exception as e:
    print(f"[ERROR] Failed to load ML model: {e}")
    rf_model = None
    scaler = None
    feature_columns = None


def _build_feature_vector(data: dict) -> np.ndarray:
    """
    Build a feature vector from user-submitted profile data.
    Maps incoming API fields to the model's expected features.
    """
    # Parse incoming data with defaults
    profile_pic = 1 if data.get("profile_pic", True) else 0
    num_followers = float(data.get("num_followers", 0))
    num_following = float(data.get("num_following", 0))
    num_posts = float(data.get("num_posts", 0))
    desc_length = float(data.get("desc_length", 0))
    external_url = 1 if data.get("external_url", False) else 0
    private = 1 if data.get("private", False) else 0

    # Username analysis
    username = data.get("username", "")
    if username:
        digit_count = sum(1 for c in username if c.isdigit())
        ratio_num_username = digit_count / len(username) if username else 0
    else:
        ratio_num_username = float(data.get("ratio_num_username", 0))

    # Fullname analysis
    fullname = data.get("fullname", "")
    if fullname:
        fullname_words = len(fullname.split())
        digit_count_fn = sum(1 for c in fullname if c.isdigit())
        ratio_num_fullname = digit_count_fn / len(fullname) if fullname else 0
    else:
        fullname_words = int(data.get("fullname_words", 1))
        ratio_num_fullname = float(data.get("ratio_num_fullname", 0))

    # Name == Username match
    name_eq_username = 0
    if username and fullname:
        if username.lower() == fullname.lower().replace(" ", ""):
            name_eq_username = 1

    # Bio length from bio text
    bio = data.get("bio", "")
    if bio and desc_length == 0:
        desc_length = len(bio)

    # Engineered features
    followers_following_ratio = num_followers / (num_following + 1)
    posts_per_follower = num_posts / (num_followers + 1)
    has_bio = 1 if desc_length > 0 else 0
    high_digit_username = 1 if ratio_num_username > 0.3 else 0

    # Build feature vector in the same order as training
    features = [
        profile_pic, ratio_num_username, fullname_words,
        ratio_num_fullname, name_eq_username, desc_length,
        external_url, private, num_posts, num_followers,
        num_following, followers_following_ratio,
        posts_per_follower, has_bio, high_digit_username,
    ]

    return np.array(features).reshape(1, -1)


def _explain_prediction(data: dict, prediction: int, probabilities: list) -> list:
    """
    Generate human-readable explanations for why a profile
    is classified as fake or real.
    """
    reasons = []
    fake_prob = probabilities[1]

    num_followers = float(data.get("num_followers", 0))
    num_following = float(data.get("num_following", 0))
    num_posts = float(data.get("num_posts", 0))
    profile_pic = data.get("profile_pic", True)
    bio = data.get("bio", "")
    username = data.get("username", "")
    desc_length = float(data.get("desc_length", len(bio)))

    if prediction == 1:  # Fake
        if not profile_pic:
            reasons.append("❌ No profile picture — common in fake accounts")
        if num_followers < 50:
            reasons.append(f"❌ Very low follower count ({int(num_followers)})")
        if num_following > 0 and num_followers / (num_following + 1) < 0.1:
            reasons.append("❌ Extremely low followers/following ratio — follows many but few follow back")
        if num_posts == 0:
            reasons.append("❌ Zero posts — inactive or bot account")
        elif num_posts < 5:
            reasons.append(f"❌ Very few posts ({int(num_posts)}) — suspicious activity level")
        if desc_length == 0:
            reasons.append("❌ Empty bio/description")
        if username:
            digit_ratio = sum(1 for c in username if c.isdigit()) / len(username)
            if digit_ratio > 0.3:
                reasons.append(f"❌ Username has high digit ratio ({digit_ratio:.0%}) — auto-generated pattern")
        if num_following > 1000 and num_followers < 100:
            reasons.append("❌ Mass following with few followers — typical bot behavior")
    else:  # Real
        if profile_pic:
            reasons.append("✅ Has profile picture")
        if num_followers > 100:
            reasons.append(f"✅ Healthy follower count ({int(num_followers)})")
        if num_posts > 10:
            reasons.append(f"✅ Active posting history ({int(num_posts)} posts)")
        if desc_length > 0:
            reasons.append("✅ Has bio/description")
        if num_following > 0:
            ratio = num_followers / (num_following + 1)
            if 0.1 < ratio < 10:
                reasons.append("✅ Normal followers/following ratio")

    if not reasons:
        reasons.append("Analysis based on overall feature pattern matching")

    return reasons


def _apply_image_heuristics(url: str, is_fake_profile: bool, profile_proba: list, all_reasons: list) -> tuple:
    """Applies heuristic rule adjustments based on image module results."""
    image_result = analyze_image(url)
    
    if not url or (image_result["has_face"] == 0 and image_result["same_image_reuse"] == 0):
        return image_result, is_fake_profile, profile_proba
        
    adjusted_fake_prob = profile_proba[1]
    
    if image_result["same_image_reuse"] == 1:
        adjusted_fake_prob = min(0.99, adjusted_fake_prob + 0.35)
        all_reasons.append("🚨 Exact duplicate profile picture detected (strong indicator of fake account)")
    
    if url and image_result["has_face"] == 0:
        adjusted_fake_prob = min(0.99, adjusted_fake_prob + 0.05)
        all_reasons.append("⚠️ Profile picture contains no human face")

    # Re-evaluate final fake boolean based on new probability
    final_is_fake = bool(adjusted_fake_prob >= 0.5)
    
    # Adjust real probability
    adjusted_real_prob = 1.0 - adjusted_fake_prob
    
    return image_result, final_is_fake, [adjusted_real_prob, adjusted_fake_prob]


# ══════════════════════════════════════════════════════════════
#  BEHAVIOR HELPER
# ══════════════════════════════════════════════════════════════

def _run_behavior_analysis(profile_data: dict, posts: list,
                           fake_prob: float, reasons: list) -> tuple:
    """
    Run behavior_module analysis and apply rule-based probability adjustments.
    Returns (behavior_result, adjusted_fake_prob, updated_reasons).
    """
    try:
        behavior = analyze_behavior(profile_data, posts)
        adjusted_prob, reasons = apply_behavior_adjustments(
            behavior, fake_prob, reasons
        )
        return behavior, adjusted_prob, reasons
    except Exception as e:
        print(f"[BEHAVIOR] Analysis error: {e}")
        return {}, fake_prob, reasons


# ══════════════════════════════════════════════════════════════
#  API ENDPOINTS
# ══════════════════════════════════════════════════════════════

@api_view(["POST"])
def predict_profile(request):
    """
    POST /api/predict-profile/
    Predict whether a social media profile is fake or real.

    Expected JSON body:
    {
        "username": "john_doe123",
        "fullname": "John Doe",
        "bio": "Some bio text",
        "num_followers": 500,
        "num_following": 200,
        "num_posts": 50,
        "profile_pic": true,
        "external_url": false,
        "private": false
    }
    """
    if rf_model is None:
        return Response(
            {"error": "ML model not loaded. Please train the model first."},
            status=status.HTTP_503_SERVICE_UNAVAILABLE,
        )

    data = request.data

    try:
        features = _build_feature_vector(data)
        features_scaled = scaler.transform(features)

        prediction = rf_model.predict(features_scaled)[0]
        probabilities = rf_model.predict_proba(features_scaled)[0].tolist()

        reasons = _explain_prediction(data, prediction, probabilities)

        # Apply image heuristics
        profile_pic_url = data.get("profile_pic_url", "")
        is_fake_profile = bool(prediction == 1)
        image_result, is_fake_profile, probabilities = _apply_image_heuristics(
            profile_pic_url, is_fake_profile, probabilities, reasons
        )

        result = {
            "prediction": "FAKE" if is_fake_profile else "REAL",
            "is_fake": is_fake_profile,
            "confidence": round(max(probabilities) * 100, 1),
            "fake_probability": round(probabilities[1] * 100, 1),
            "real_probability": round(probabilities[0] * 100, 1),
            "reasons": reasons,
            "image_analysis": image_result,
        }

        return Response(result, status=status.HTTP_200_OK)

    except Exception as e:
        return Response(
            {"error": f"Prediction failed: {str(e)}"},
            status=status.HTTP_400_BAD_REQUEST,
        )


@api_view(["POST"])
def detect_ai_text_view(request):
    """
    POST /api/detect-ai-text/
    Detect if text is AI-generated.

    Expected JSON body:
    {
        "text": "Some text to analyze..."
    }
    """
    text = request.data.get("text", "")

    if not text:
        return Response(
            {"error": "No text provided. Send 'text' in JSON body."},
            status=status.HTTP_400_BAD_REQUEST,
        )

    result = detect_ai_text(text)
    result["confidence_percent"] = round(result["confidence"] * 100, 1)

    return Response(result, status=status.HTTP_200_OK)


@api_view(["POST"])
def analyze(request):
    """
    POST /api/analyze/
    Combined analysis: profile features + AI text detection.

    Logic: IF fake_profile OR ai_generated_content → mark as FAKE

    Expected JSON body (combines both endpoints):
    {
        "username": "john_doe123",
        "fullname": "John Doe",
        "bio": "Passionate about leveraging...",
        "num_followers": 500,
        "num_following": 200,
        "num_posts": 50,
        "profile_pic": true,
        "external_url": false,
        "private": false
    }
    """
    if rf_model is None:
        return Response(
            {"error": "ML model not loaded."},
            status=status.HTTP_503_SERVICE_UNAVAILABLE,
        )

    data = request.data

    try:
        # ── 1. Profile prediction ────────────────────────────
        features = _build_feature_vector(data)
        features_scaled = scaler.transform(features)

        profile_pred = rf_model.predict(features_scaled)[0]
        profile_proba = rf_model.predict_proba(features_scaled)[0].tolist()
        profile_reasons = _explain_prediction(data, profile_pred, profile_proba)

        profile_pic_url = data.get("profile_pic_url", "")
        is_fake_profile = bool(profile_pred == 1)
        image_result, is_fake_profile, profile_proba = _apply_image_heuristics(
            profile_pic_url, is_fake_profile, profile_proba, profile_reasons
        )

        # ── 2. AI text detection on bio ──────────────────────
        bio = data.get("bio", "")
        ai_result = detect_ai_text(bio) if bio else {
            "is_ai_generated": False,
            "confidence": 0.0,
            "reasons": ["No bio text provided"],
            "scores": {},
        }

        # ── 3. Combined verdict ──────────────────────────────
        # IF fake profile OR AI-generated content → FAKE
        # is_fake_profile already adjusted via image heuristics
        is_ai_content = ai_result["is_ai_generated"]

        final_fake = is_fake_profile or is_ai_content

        # Build combined reasons
        all_reasons = []
        if is_fake_profile:
            all_reasons.append("⚠️ Profile features indicate FAKE account")
            all_reasons.extend(profile_reasons)
        if is_ai_content:
            all_reasons.append("⚠️ Bio text appears to be AI-generated")
            all_reasons.extend(ai_result["reasons"])

        if not final_fake:
            all_reasons.append("✅ Profile appears genuine")
            all_reasons.extend(profile_reasons)
            if bio:
                all_reasons.append("✅ Bio text appears human-written")

        result = {
            "final_verdict": "FAKE" if final_fake else "REAL",
            "is_fake": final_fake,
            "confidence": round(
                max(profile_proba[1] if is_fake_profile else profile_proba[0],
                    ai_result["confidence"] if is_ai_content else 1 - ai_result["confidence"])
                * 100, 1
            ),
            "profile_analysis": {
                "prediction": "FAKE" if is_fake_profile else "REAL",
                "fake_probability": round(profile_proba[1] * 100, 1),
                "real_probability": round(profile_proba[0] * 100, 1),
                "reasons": profile_reasons,
            },
            "image_analysis": image_result,
            "ai_text_analysis": {
                "is_ai_generated": is_ai_content,
                "confidence": round(ai_result["confidence"] * 100, 1),
                "reasons": ai_result["reasons"],
                "scores": ai_result.get("scores", {}),
            },
            "reasons": all_reasons,
        }

        return Response(result, status=status.HTTP_200_OK)

    except Exception as e:
        return Response(
            {"error": f"Analysis failed: {str(e)}"},
            status=status.HTTP_400_BAD_REQUEST,
        )


@api_view(["POST"])
def analyze_posts(request):
    """
    POST /api/analyze-posts/
    Full multi-signal behavior + content analysis on a list of posts.

    Expected JSON body:
    {
        "profile": {
            "username": "john_doe",
            "bio": "Entrepreneur | Investor",
            "num_followers": 1200,
            "num_following": 800,
            "num_posts": 45,
            "profile_pic": true,
            "profile_pic_url": "https://..."
        },
        "posts": [
            {
                "timestamp": "2024-01-15T14:30:00",
                "caption": "Earn money fast! Link in bio!",
                "likes": 5,
                "media_type": "image",
                "media_url": "https://..."
            }
        ]
    }
    """
    if rf_model is None:
        return Response(
            {"error": "ML model not loaded."},
            status=status.HTTP_503_SERVICE_UNAVAILABLE,
        )

    data = request.data
    profile_data = data.get("profile", {})
    posts = data.get("posts", [])

    if not isinstance(posts, list):
        return Response(
            {"error": "'posts' must be a JSON array."},
            status=status.HTTP_400_BAD_REQUEST,
        )

    try:
        # ── 1. Profile ML prediction ──────────────────────────
        features = _build_feature_vector(profile_data)
        features_scaled = scaler.transform(features)
        profile_pred = rf_model.predict(features_scaled)[0]
        profile_proba = rf_model.predict_proba(features_scaled)[0].tolist()
        profile_reasons = _explain_prediction(profile_data, profile_pred, profile_proba)

        # ── 2. Image heuristics on profile picture ────────────
        profile_pic_url = profile_data.get("profile_pic_url", "")
        is_fake_profile = bool(profile_pred == 1)
        image_result, is_fake_profile, profile_proba = _apply_image_heuristics(
            profile_pic_url, is_fake_profile, profile_proba, profile_reasons
        )

        # ── 3. AI text detection on bio ───────────────────────
        bio = profile_data.get("bio", "")
        ai_result = detect_ai_text(bio) if bio else {
            "is_ai_generated": False, "confidence": 0.0,
            "reasons": ["No bio text provided"], "scores": {},
        }

        # ── 4. Behavior / content / video analysis ────────────
        all_reasons = list(profile_reasons)
        behavior_result, adjusted_fake_prob, all_reasons = _run_behavior_analysis(
            profile_data, posts, profile_proba[1], all_reasons
        )

        # ── 5. Final verdict ──────────────────────────────────
        is_ai_content = ai_result["is_ai_generated"]
        final_fake = (adjusted_fake_prob >= 0.5) or is_ai_content

        final_reasons = []
        if final_fake:
            if adjusted_fake_prob >= 0.5:
                final_reasons.append("⚠️ Profile + behavior signals indicate FAKE account")
            if is_ai_content:
                final_reasons.append("⚠️ Bio text appears to be AI-generated")
                final_reasons.extend(ai_result["reasons"])
        else:
            final_reasons.append("✅ Profile appears genuine based on all signals")
        final_reasons.extend(all_reasons)

        # Behavior-specific reasons (from behavior module itself)
        b_reasons = behavior_result.get("reasons", [])
        for r in b_reasons:
            if r not in final_reasons:
                final_reasons.append(r)

        real_prob = round(1.0 - adjusted_fake_prob, 3)

        result = {
            "final_verdict": "FAKE" if final_fake else "REAL",
            "is_fake": final_fake,
            "confidence": round(max(adjusted_fake_prob, 1.0 - adjusted_fake_prob) * 100, 1),
            "fake_probability": round(adjusted_fake_prob * 100, 1),
            "real_probability": round(real_prob * 100, 1),
            "profile_analysis": {
                "prediction": "FAKE" if is_fake_profile else "REAL",
                "fake_probability": round(profile_proba[1] * 100, 1),
                "real_probability": round(profile_proba[0] * 100, 1),
                "reasons": profile_reasons,
            },
            "image_analysis": image_result,
            "ai_text_analysis": {
                "is_ai_generated": is_ai_content,
                "confidence": round(ai_result["confidence"] * 100, 1),
                "reasons": ai_result["reasons"],
            },
            "behavior_analysis": {
                "posts_per_day": behavior_result.get("posts_per_day", 0),
                "time_variance_score": behavior_result.get("time_variance_score", 0),
                "avg_time_gap_hours": behavior_result.get("avg_time_gap_hours", 0),
                "engagement_ratio": behavior_result.get("engagement_ratio", 0),
                "duplicate_text_ratio": behavior_result.get("duplicate_text_ratio", 0),
                "lexical_diversity": behavior_result.get("lexical_diversity", 1),
                "ai_text_score": behavior_result.get("ai_text_score", 0),
                "agenda_score": behavior_result.get("agenda_score", 0),
                "content_similarity_score": behavior_result.get("content_similarity_score", 0),
                "face_presence_ratio": behavior_result.get("face_presence_ratio", 0),
                "image_reuse_detected": behavior_result.get("image_reuse_detected", False),
                "video_transcripts_count": behavior_result.get("video_transcripts_count", 0),
                "ffmpeg_available": behavior_result.get("ffmpeg_available", False),
                "whisper_available": behavior_result.get("whisper_available", False),
                "posts_analyzed": behavior_result.get("post_count_analyzed", 0),
                "reasons": behavior_result.get("reasons", []),
            },
            "reasons": final_reasons,
        }

        return Response(result, status=status.HTTP_200_OK)

    except Exception as e:
        return Response(
            {"error": f"Post analysis failed: {str(e)}"},
            status=status.HTTP_400_BAD_REQUEST,
        )


@api_view(["POST"])
def analyze_url(request):
    """
    POST /api/analyze-url/
    Scrape Instagram profile from URL, then run combined analysis.

    Expected JSON body:
    {
        "url": "https://instagram.com/username"
    }
    """
    if rf_model is None:
        return Response(
            {"error": "ML model not loaded."},
            status=status.HTTP_503_SERVICE_UNAVAILABLE,
        )

    url = request.data.get("url", "").strip()
    if not url:
        return Response(
            {"error": "No URL provided. Send 'url' in JSON body."},
            status=status.HTTP_400_BAD_REQUEST,
        )

    # ── 1. Scrape profile data ────────────────────────────────
    scraped = scrape_instagram_profile(url)

    if not scraped.get("success"):
        return Response(
            {"error": scraped.get("error", "Failed to fetch profile.")},
            status=status.HTTP_400_BAD_REQUEST,
        )

    # ── 2. Run profile prediction ─────────────────────────────
    data = scraped  # scraped dict has all the fields we need
    try:
        features = _build_feature_vector(data)
        features_scaled = scaler.transform(features)

        profile_pred = rf_model.predict(features_scaled)[0]
        profile_proba = rf_model.predict_proba(features_scaled)[0].tolist()
        profile_reasons = _explain_prediction(data, profile_pred, profile_proba)

        profile_pic_url = data.get("profile_pic_url", "")
        is_fake_profile = bool(profile_pred == 1)
        image_result, is_fake_profile, profile_proba = _apply_image_heuristics(
            profile_pic_url, is_fake_profile, profile_proba, profile_reasons
        )

        # ── 3. AI text detection on bio ───────────────────────
        bio = data.get("bio", "")
        ai_result = detect_ai_text(bio) if bio else {
            "is_ai_generated": False,
            "confidence": 0.0,
            "reasons": ["No bio text provided"],
            "scores": {},
        }

        # ── 4. Combined verdict ───────────────────────────────
        # is_fake_profile corresponds to image heuristics adjusted value
        is_ai_content = ai_result["is_ai_generated"]
        final_fake = is_fake_profile or is_ai_content

        all_reasons = []
        if is_fake_profile:
            all_reasons.append("⚠️ Profile features indicate FAKE account")
            all_reasons.extend(profile_reasons)
        if is_ai_content:
            all_reasons.append("⚠️ Bio text appears to be AI-generated")
            all_reasons.extend(ai_result["reasons"])
        if not final_fake:
            all_reasons.append("✅ Profile appears genuine")
            all_reasons.extend(profile_reasons)
            if bio:
                all_reasons.append("✅ Bio text appears human-written")

        result = {
            "final_verdict": "FAKE" if final_fake else "REAL",
            "is_fake": final_fake,
            "confidence": round(
                max(profile_proba[1] if is_fake_profile else profile_proba[0],
                    ai_result["confidence"] if is_ai_content else 1 - ai_result["confidence"])
                * 100, 1
            ),
            "scraped_profile": {
                "username": data.get("username", ""),
                "fullname": data.get("fullname", ""),
                "bio": bio,
                "num_followers": data.get("num_followers", 0),
                "num_following": data.get("num_following", 0),
                "num_posts": data.get("num_posts", 0),
                "profile_pic": data.get("profile_pic", False),
                "external_url": data.get("external_url", False),
                "private": data.get("private", False),
                "is_verified": data.get("is_verified", False),
            },
            "profile_analysis": {
                "prediction": "FAKE" if is_fake_profile else "REAL",
                "fake_probability": round(profile_proba[1] * 100, 1),
                "real_probability": round(profile_proba[0] * 100, 1),
                "reasons": profile_reasons,
            },
            "image_analysis": image_result,
            "ai_text_analysis": {
                "is_ai_generated": is_ai_content,
                "confidence": round(ai_result["confidence"] * 100, 1),
                "reasons": ai_result["reasons"],
                "scores": ai_result.get("scores", {}),
            },
            "reasons": all_reasons,
        }

        return Response(result, status=status.HTTP_200_OK)

    except Exception as e:
        return Response(
            {"error": f"Analysis failed: {str(e)}"},
            status=status.HTTP_400_BAD_REQUEST,
        )
