# AI-Based Fake Social Media Profile Detection System

## Project Report

---

## 1. Introduction

### 1.1 Problem Statement

Social media platforms are increasingly plagued by fake accounts that spread misinformation, conduct phishing attacks, run scam campaigns, and artificially inflate follower counts. Instagram alone removes millions of fake accounts every quarter. Detecting these accounts manually is impractical at scale, necessitating automated solutions powered by machine learning.

### 1.2 Objective

This project aims to build an **end-to-end AI-based system** that:

1. **Classifies social media profiles** as fake or genuine using a Random Forest machine learning model trained on Instagram profile features.
2. **Detects AI-generated content** in profile bios and posts using NLP-based heuristic analysis.
3. **Provides explainability** — the system doesn't just give a verdict, but explains *why* a profile is suspicious.
4. **Exposes a REST API** backend and a modern **web-based frontend** for interactive analysis.

### 1.3 Scope

- Platform focus: **Instagram** (architecture easily extendable to Twitter/X, Facebook, etc.)
- Detection methods: Profile feature analysis + AI-generated text detection
- Deployment: Local development server with Django REST Framework

---

## 2. Literature Review / Background

| Approach | Description | Limitation |
|---|---|---|
| Rule-based | Hardcoded checks (e.g., no profile pic = fake) | Low accuracy, easily evaded |
| Traditional ML | SVM, Decision Trees, Random Forest on profile features | Requires feature engineering |
| Deep Learning | CNNs on profile images, RNNs on post text | Heavy computation, needs GPU |
| Graph-based | Social network graph analysis (follower/following patterns) | Requires access to graph data |

This project uses **Traditional ML (Random Forest)** for profile classification combined with **heuristic NLP** for AI text detection — balancing accuracy with portability and speed.

---

## 3. System Architecture

```
┌─────────────────────────────┐
│        Frontend (UI)        │
│  HTML / CSS / JavaScript    │
│  Glassmorphism Dark Theme   │
└──────────┬──────────────────┘
           │  HTTP (JSON)
           ▼
┌─────────────────────────────┐
│    Django REST Framework    │
│    Backend (API Server)     │
├─────────────────────────────┤
│  /api/analyze-url/          │ ← URL-based auto-fetch
│  /api/analyze/              │ ← Combined analysis
│  /api/predict-profile/      │ ← Profile ML prediction
│  /api/detect-ai-text/       │ ← AI text detection
└──────┬──────────┬───────────┘
       │          │
       ▼          ▼
┌────────────┐ ┌──────────────┐
│  Random    │ │ AI Text      │
│  Forest    │ │ Detector     │
│  Model     │ │ (Heuristic)  │
│  (.pkl)    │ │              │
└────────────┘ └──────────────┘
       │
       ▼
┌────────────────────────────┐
│     Instagram Scraper       │
│  (instaloader library)      │
│  Auto-fetch profile data    │
└────────────────────────────┘
```

---

## 4. Dataset

### 4.1 Source

**Instagram Fake & Spammer Genuine Accounts Dataset** (`train.csv`)

- **Records:** 576 Instagram profiles
- **Label:** `fake` (1 = fake, 0 = genuine)
- **Fake ratio:** ~50% (balanced dataset)

### 4.2 Original Features (11 columns)

| # | Feature | Type | Description |
|---|---------|------|-------------|
| 1 | `profile_pic` | Binary | Has a profile picture (1) or not (0) |
| 2 | `nums/length_username` | Float | Ratio of digits in username |
| 3 | `fullname_words` | Int | Number of words in full name |
| 4 | `nums/length_fullname` | Float | Ratio of digits in full name |
| 5 | `name==username` | Binary | Full name matches username |
| 6 | `description_length` | Int | Length of profile bio |
| 7 | `external_URL` | Binary | Has external URL |
| 8 | `private` | Binary | Account is private |
| 9 | `#posts` | Int | Number of posts |
| 10 | `#followers` | Int | Follower count |
| 11 | `#follows` | Int | Following count |

### 4.3 Engineered Features (4 additional)

| Feature | Formula | Rationale |
|---------|---------|-----------|
| `followers_following_ratio` | followers / (following + 1) | Bots follow many but have few followers |
| `posts_per_follower` | posts / (followers + 1) | Engagement indicator |
| `has_bio` | 1 if desc_length > 0 | Fake accounts often skip bios |
| `high_digit_username` | 1 if digit_ratio > 0.3 | Auto-generated usernames have many digits |

**Total features used for training: 15**

---

## 5. Methodology

### 5.1 Machine Learning Pipeline

```
Raw CSV Data
    │
    ▼
Data Preprocessing
  • Column rename, null handling
  • Feature engineering (4 new features)
    │
    ▼
Train/Test Split (80/20, stratified)
    │
    ▼
Feature Scaling (StandardScaler)
    │
    ▼
Random Forest Training
  • n_estimators = 100
  • max_depth = 15
  • min_samples_split = 5
  • min_samples_leaf = 2
    │
    ▼
Model Evaluation
  • Accuracy, Confusion Matrix
  • ROC-AUC, Classification Report
    │
    ▼
Model Serialization (joblib)
  • fake_profile_model.pkl
  • scaler.pkl
  • feature_columns.pkl
```

### 5.2 AI Text Detection Pipeline

Instead of heavy transformer models (like RoBERTa), a **lightweight heuristic approach** was implemented for portability:

| Heuristic | Weight | Logic |
|-----------|--------|-------|
| AI Phrase Density | 25% | Counts known AI-typical buzzwords ("leverage", "foster", "delve", etc.) |
| Sentence Uniformity | 20% | Low coefficient of variation in sentence lengths = suspicious |
| Vocabulary Diversity | 15% | Low Type-Token Ratio = repetitive AI vocabulary |
| Generic Patterns | 15% | Regex matching formulaic bio patterns ("passionate about...") |
| Repetitive Structure | 15% | Repetitive sentence openings |
| Punctuation Regularity | 10% | Suspiciously consistent punctuation patterns |

**Decision threshold:** confidence ≥ 0.4 → classified as AI-generated

### 5.3 Combined Verdict Logic

```
IF (profile_prediction == FAKE) OR (bio_is_ai_generated == TRUE):
    final_verdict = "FAKE"
ELSE:
    final_verdict = "REAL"
```

This "fail-fast" approach ensures suspicious signals from either module trigger a FAKE verdict.

---

## 6. Results

### 6.1 Random Forest Model Performance

| Metric | Value |
|--------|-------|
| **Accuracy** | **90.52%** |
| **ROC-AUC** | **0.9819** |
| **Precision (Fake)** | 89% |
| **Recall (Fake)** | 93% |
| **Precision (Real)** | 92% |
| **Recall (Real)** | 88% |

### 6.2 Confusion Matrix

```
              Predicted
              Real    Fake
Actual Real    51       7
Actual Fake     4      54
```

- **True Positives (TP):** 54 — correctly identified fake profiles
- **True Negatives (TN):** 51 — correctly identified real profiles
- **False Positives (FP):** 7 — real profiles misclassified as fake
- **False Negatives (FN):** 4 — fake profiles missed

### 6.3 Feature Importance (Top 10)

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | `num_followers` | 23.4% |
| 2 | `num_posts` | 18.0% |
| 3 | `followers_following_ratio` | 9.9% |
| 4 | `profile_pic` | 9.8% |
| 5 | `posts_per_follower` | 8.5% |
| 6 | `num_following` | 8.0% |
| 7 | `desc_length` | 7.8% |
| 8 | `ratio_num_username` | 5.3% |
| 9 | `fullname_words` | 3.5% |
| 10 | `high_digit_username` | 2.3% |

**Key Insight:** Follower count is the single most important feature, followed by post count and the follower/following ratio — confirming that fake accounts have characteristically low engagement.

### 6.4 AI Text Detection — Sample Results

| Input Text | Verdict | Confidence |
|------------|---------|------------|
| "Passionate about leveraging technology to make the world a better place..." | 🤖 AI | 55.8% |
| "hey guys!! just got back from the beach lol 🏖️ pizza was amazing..." | 👤 Human | ~5% |

---

## 7. Technology Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **ML Model** | scikit-learn (RandomForestClassifier) | Profile classification |
| **Data Processing** | pandas, numpy | Feature engineering, data manipulation |
| **AI Text Detection** | Python (regex, collections) | Heuristic NLP analysis |
| **Profile Scraping** | instaloader | Auto-fetch Instagram profile data |
| **Backend API** | Django + Django REST Framework | REST API endpoints |
| **CORS** | django-cors-headers | Cross-origin request handling |
| **Frontend** | HTML5, CSS3, Vanilla JavaScript | Interactive web UI |
| **Serialization** | joblib | Model persistence (.pkl files) |
| **Visualization** | matplotlib, seaborn | Training plots |

---

## 8. API Documentation

### 8.1 `POST /api/analyze-url/`

**Purpose:** Auto-fetch Instagram profile by URL and run combined analysis.

**Request:**
```json
{
    "url": "https://instagram.com/username"
}
```

**Response:**
```json
{
    "final_verdict": "FAKE",
    "is_fake": true,
    "confidence": 91.0,
    "scraped_profile": {
        "username": "username",
        "fullname": "...",
        "bio": "...",
        "num_followers": 5,
        "num_following": 800,
        "num_posts": 0,
        "profile_pic": false
    },
    "profile_analysis": {
        "prediction": "FAKE",
        "fake_probability": 91.0,
        "reasons": ["❌ No profile picture", "..."]
    },
    "ai_text_analysis": {
        "is_ai_generated": true,
        "confidence": 55.8,
        "reasons": ["Contains AI-typical phrases"]
    },
    "reasons": ["⚠️ Profile features indicate FAKE account", "..."]
}
```

### 8.2 `POST /api/analyze/`

**Purpose:** Combined analysis from manually entered profile data.

### 8.3 `POST /api/predict-profile/`

**Purpose:** Profile feature analysis only (ML model prediction).

### 8.4 `POST /api/detect-ai-text/`

**Purpose:** AI-generated text detection only.

---

## 9. Project Structure

```
fake_profile/
├── ml_model/
│   ├── train_model.py              # Model training script
│   ├── ai_text_detector.py         # AI text heuristic detector
│   ├── profile_scraper.py          # Instagram URL scraper
│   ├── fake_profile_model.pkl      # Trained Random Forest model
│   ├── scaler.pkl                  # Fitted StandardScaler
│   ├── feature_columns.pkl         # Feature column names
│   ├── feature_importances.png     # Feature importance chart
│   └── confusion_matrix.png        # Confusion matrix heatmap
├── backend/
│   ├── manage.py                   # Django management
│   ├── backend/
│   │   ├── settings.py             # Django configuration
│   │   └── urls.py                 # Root URL routing
│   └── api/
│       ├── views.py                # 4 API endpoints
│       └── urls.py                 # API URL routes
├── frontend/
│   ├── index.html                  # Main UI page
│   ├── style.css                   # Dark glassmorphism design
│   └── script.js                   # API calls & rendering
├── instagram-fake-spammer-genuine-accounts/
│   └── train.csv                   # Training dataset
└── PROJECT_REPORT.md               # This report
```

---

## 10. How to Run

```bash
# 1. Install dependencies
pip install scikit-learn pandas numpy matplotlib seaborn joblib
pip install django djangorestframework django-cors-headers
pip install instaloader

# 2. Train the model (optional — pre-trained model included)
cd /path/to/fake_profile
python3 ml_model/train_model.py

# 3. Run Django migrations
cd backend
python3 manage.py migrate

# 4. Start the server
python3 manage.py runserver 8000

# 5. Open in browser
# Navigate to http://127.0.0.1:8000/
```

---

## 11. Explainability

A critical design goal was **transparency in predictions**. The system provides human-readable explanations for every verdict:

**For FAKE profiles:**
- ❌ No profile picture — common in fake accounts
- ❌ Very low follower count (5)
- ❌ Extremely low followers/following ratio — follows many but few follow back
- ❌ Zero posts — inactive or bot account
- ❌ Username has high digit ratio (36%) — auto-generated pattern

**For ai-generated text:**
- Contains multiple AI-typical phrases and buzzwords
- Uses generic/formulaic patterns common in AI output
- Suspiciously uniform sentence lengths

---

## 12. Limitations & Future Work

### 12.1 Current Limitations

| Limitation | Impact |
|------------|--------|
| Instagram rate-limiting | Scraping may fail for frequent requests |
| Small dataset (576 samples) | Model may not generalize to all edge cases |
| No image analysis | Profile picture authenticity not checked |
| Heuristic text detection | Less accurate than transformer-based models |
| Single platform | Currently only supports Instagram URLs |

### 12.2 Future Enhancements

1. **Image Analysis** — Use CNN models to detect stock photos, GAN-generated faces, and stolen images.
2. **Transformer-based text detection** — Integrate RoBERTa or GPT-based classifiers for higher accuracy AI text detection.
3. **Larger dataset** — Train on datasets with 50K+ profiles for better generalization.
4. **Multi-platform support** — Extend URL scraping to Twitter/X, Facebook, LinkedIn.
5. **Real-time monitoring** — Deploy as a browser extension for automatic profile checking.
6. **Production deployment** — Containerize with Docker, deploy on AWS/GCP with Gunicorn + Nginx.

---

## 13. Conclusion

This project successfully demonstrates an end-to-end AI-based system for detecting fake social media profiles. The Random Forest model achieves **90.52% accuracy** with a **0.98 ROC-AUC**, effectively identifying fake accounts based on structural profile features. The AI text detection module complements this by flagging AI-generated bios. The combined "fail-fast" logic — where either a suspicious profile structure OR AI-generated content triggers a FAKE verdict — provides a robust multi-signal detection approach.

The system's key strength lies in its **explainability**: rather than providing a black-box prediction, it clearly communicates *why* a profile is suspicious, making it both useful for end-users and valuable for academic research.

---

## 14. References

1. Cresci, S. (2020). A Decade of Social Bot Detection. *Communications of the ACM*.
2. Instagram. (2023). How We Fight Inauthentic Behavior. *Meta Transparency Center*.
3. scikit-learn documentation. RandomForestClassifier. https://scikit-learn.org/stable/
4. Django REST Framework documentation. https://www.django-rest-framework.org/
5. Instaloader documentation. https://instaloader.github.io/

---

*Report generated on: March 26, 2026*
*Project by: Nishant Mishra*
