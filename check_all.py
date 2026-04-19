"""
Diagnostic script — run this to check ALL components work.
Usage:
  cd d:\Downloads\fake_profile\fake_profile\backend
  d:\Downloads\fake_profile\fake_profile\venv_313\Scripts\python.exe ..\check_all.py
"""
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'backend.settings')

PASS = "[  OK  ]"
FAIL = "[ FAIL ]"

def check(label, fn):
    try:
        fn()
        print(f"{PASS} {label}")
        return True
    except Exception as e:
        print(f"{FAIL} {label}")
        print(f"         Error: {e}")
        return False

print("\n" + "="*55)
print("  Fake Profile Detector — Full Diagnostic Check")
print("="*55 + "\n")

# 1. Django
def t1():
    import django
    django.setup()
check("Django setup", t1)

# 2. ML model files
def t2():
    from django.conf import settings
    import joblib
    joblib.load(os.path.join(settings.ML_MODEL_DIR, "fake_profile_model.pkl"))
    joblib.load(os.path.join(settings.ML_MODEL_DIR, "scaler.pkl"))
    joblib.load(os.path.join(settings.ML_MODEL_DIR, "feature_columns.pkl"))
check("ML model files (pkl)", t2)

# 3. behavior_module
def t3():
    from ml_model.behavior_module import analyze_behavior, apply_behavior_adjustments
    result = analyze_behavior(
        {"bio": "Test bio", "num_followers": 100},
        [{"timestamp": "2024-01-01T10:00:00", "caption": "Hello world", "likes": 5, "media_type": "image", "media_url": ""}]
    )
    assert "posts_per_day" in result
check("behavior_module imports + basic run", t3)

# 4. image_module
def t4():
    from ml_model.image_module import analyze_image
check("image_module import", t4)

# 5. ai_text_detector
def t5():
    from ml_model.ai_text_detector import detect_ai_text
    r = detect_ai_text("I am a passionate entrepreneur leveraging synergy.")
    assert "is_ai_generated" in r
check("ai_text_detector import + run", t5)

# 6. views.py analyze_posts
def t6():
    from api.views import analyze_posts
check("views.py analyze_posts function", t6)

# 7. urls registered
def t7():
    from django.urls import reverse
    reverse("analyze-posts")
check("URL route /api/analyze-posts/ registered", t7)

# 8. Whisper
def t8():
    import whisper
    print(f"\n         (whisper version: {whisper.__version__})", end="")
check("openai-whisper installed", t8)

# 9. FFmpeg
def t9():
    import subprocess
    r = subprocess.run(["ffmpeg", "-version"], capture_output=True, timeout=5)
    assert r.returncode == 0
check("FFmpeg on system PATH", t9)

# 10. sklearn
def t10():
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
check("scikit-learn TF-IDF available", t10)

# 11. OpenCV + ImageHash
def t11():
    import cv2, imagehash
check("OpenCV + ImageHash available", t11)

print("\n" + "="*55)
print("  Diagnostic complete. Fix any [ FAIL ] items above.")
print("="*55 + "\n")
