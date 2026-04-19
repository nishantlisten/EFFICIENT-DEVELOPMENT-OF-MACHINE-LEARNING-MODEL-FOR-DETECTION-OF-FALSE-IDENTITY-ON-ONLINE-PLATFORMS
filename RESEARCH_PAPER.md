# AI-Driven Detection of Inauthentic Social Media Profiles: A Multi-Signal Hybrid System Combining Machine Learning, NLP, Computer Vision, and Behavioral Analysis

**Abstract**

The proliferation of fake social media accounts continues to present an escalating threat to digital trust, enabling misinformation campaigns, financial fraud, and artificial engagement inflation. Manual detection is inefficient and unscalable, necessitating automated, explainable solutions. This paper presents an end-to-end artificial intelligence system targeting Instagram profiles that has evolved from a dual-layer classifier into a four-layer multi-signal detection pipeline. Our methodology combines: (1) a Random Forest ML classifier analyzing structural profile metadata, (2) a heuristic NLP engine detecting AI-generated textual content, (3) a Computer Vision module analyzing profile images for face presence and duplicate image reuse, and (4) a novel Behavioral Consistency Module (BCM) performing temporal analysis, cross-content TF-IDF similarity, agenda detection, video/audio transcription via OpenAI Whisper, and engagement ratio analysis. Trained on 576 labeled profiles, the core Random Forest classifier achieves 90.52% accuracy and ROC-AUC of 0.98. The BCM applies rule-based probability adjustments on top of ML predictions, requiring no retraining. The combined system detects sophisticated fake accounts that evade any single detection layer by failing on behavioral or content-consistency grounds.

---

## 1. Introduction

Social networks like Instagram remove billions of sophisticated fake accounts annually. As bad actors utilize automation, purchased follower farms, and Large Language Models (LLMs) to generate realistic profiles, detection mechanisms must evolve past static, metadata-only rules. Single-signal detection approaches—relying solely on follower counts or username patterns—are increasingly bypassed by sophisticated operators who purchase followers, generate plausible bios via LLMs, and use stock images.

The objective of this research is to build a transparent, local, and privacy-preserving multi-signal detection system that:
- Analyzes structural profile metadata via supervised ML
- Detects AI-generated text in bios and post captions
- Analyzes profile and post images for face presence and duplicate reuse
- Detects automated posting behavior through temporal pattern analysis
- Identifies scripted/agenda-driven content through cross-post NLP consistency
- Transcribes and analyzes video audio content via local speech-to-text

All components operate locally with no external API dependencies, ensuring privacy preservation and offline capability.

---

## 2. Dataset Analysis and Feature Engineering

### 2.1 Base Features
The system uses the "Instagram Fake & Spammer Genuine Accounts" dataset of 576 labeled instances (~50% fake, ~50% genuine). The raw dataset provides 11 foundational metadata features:
- **Binary Features:** Profile picture presence, external URL, account privacy, full name matching username
- **Continuous/Integer Features:** Follower count, following count, post count, username length, fullname length, bio description length
- **Ratio Features:** Digit density in username and full name strings

### 2.2 Engineered Features (RF Model)
Four synthetic features capture behavioral nuances:
1. **`followers_following_ratio`** = `followers / (following + 1)` — Bots mass-follow users expecting follow-backs, producing extremely low ratios
2. **`posts_per_follower`** = `posts / (followers + 1)` — Indicates real engagement accumulation over time
3. **`has_bio`** — Binary flag from bio length; automated farms skip localized bios to save time
4. **`high_digit_username`** — Binary flag if digits exceed 30% of username (e.g., `john_doe_99381`)

### 2.3 Behavioral Post Features (BCM)
The new Behavioral Consistency Module derives 8 additional signals from post-level data:
1. **`posts_per_day`** — Average posting rate computed from post timestamps
2. **`time_variance_score`** — 1 − (std of posting hour / 12); bots post at identical clock times
3. **`engagement_ratio`** — Average likes per post ÷ follower count; extremely low in fake accounts with purchased followers
4. **`duplicate_text_ratio`** — Average pairwise TF-IDF cosine similarity across post captions
5. **`ai_text_score`** — Composite: AI phrase density × 0.4 + (1 − lexical diversity) × 0.3 + sentence uniformity × 0.3
6. **`agenda_score`** — Spam/agenda keyword density + hashtag repetition rate + top-keyword concentration
7. **`content_similarity_score`** — TF-IDF cosine similarity between bio and combined post captions
8. **`face_presence_ratio`** — Fraction of image posts where OpenCV Haar cascade detects a human face

---

## 3. System Architecture

### 3.1 Four-Layer Detection Pipeline

```
INPUT (Profile URL or Manual JSON)
         │
         ▼
┌─────────────────────────┐
│  Layer 1: Structural ML │  Random Forest on 15 features → fake_probability
└──────────┬──────────────┘
           │
           ▼
┌─────────────────────────┐
│  Layer 2: Image Vision  │  OpenCV face detection + perceptual hash reuse
└──────────┬──────────────┘
           │  ±probability adjustment
           ▼
┌─────────────────────────┐
│  Layer 3: NLP / AI Text │  Heuristic AI-phrase/diversity detector on bio
└──────────┬──────────────┘
           │  OR-gate: if AI bio → FAKE
           ▼
┌─────────────────────────┐
│  Layer 4: Behavior BCM  │  Temporal + TF-IDF + Agenda + Video/Audio
└──────────┬──────────────┘
           │  Rule-based probability bumps
           ▼
     FINAL VERDICT + EXPLANATIONS
```

### 3.2 Layer 1 — Structural Profile Classification (Random Forest)
A Random Forest ensemble classifier was selected as the primary predictive engine. Unlike deep neural networks, Random Forests are highly interpretable, resilient to overfitting on small datasets, and process tabular data with exceptional speed.
- **Training Parameters:** 100 decision trees (`n_estimators=100`), max depth 15, constrained leaf nodes to prevent overfitting
- **Scaling:** `StandardScaler` normalizes high-variance features (follower counts) before prediction
- **Output:** `fake_probability` ∈ [0, 1]

### 3.3 Layer 2 — Computer Vision (Image Module)
Implemented in `ml_model/image_module.py`:
- **Perceptual Hashing (pHash):** Detects duplicate/reused profile pictures across analyzed accounts. Hamming distance ≤ 5 triggers `same_image_reuse = 1` → +35% fake probability
- **Face Detection:** OpenCV Haar cascade (`haarcascade_frontalface_default.xml`) detects human faces. No face detected → +5% fake probability
- **Post-Level Face Ratio:** `face_presence_ratio` = fraction of image posts containing a detected human face

### 3.4 Layer 3 — AI-Generated Text Detection (NLP Module)
Implemented in `ml_model/ai_text_detector.py`. Six heuristic sub-scores are weighted:

| Heuristic | Weight | Description |
|---|---|---|
| AI Phrase Density | 25% | Overused LLM vocabulary (e.g., "leverage", "delve", "tapestry") |
| Sentence Uniformity | 20% | Low coefficient of variation in sentence lengths |
| Vocabulary Diversity | 15% | Type-Token Ratio — AI repeats vocabulary |
| Generic Patterns | 15% | Formulaic bio patterns ("passionate about", "here to inspire") |
| Repetitive Structure | 15% | Similar sentence openings across the text |
| Punctuation Regularity | 10% | Suspiciously consistent comma/period usage |

If composite score ≥ 0.4 → `is_ai_generated = True`. **Fail-Fast OR-gate:** if either structural ML OR AI text triggers → FAKE.

### 3.5 Layer 4 — Behavioral Consistency Module (BCM)
Implemented in `ml_model/behavior_module.py`. Accepts a profile dict + list of post objects. Performs:

#### 3.5.1 Text & AI Analysis Across Posts
- Concatenates all post captions for corpus-level NLP
- Computes `duplicate_text_ratio` using TF-IDF `TfidfVectorizer` + `cosine_similarity` (sklearn) on pairwise caption similarity
- Computes `ai_text_score` as composite of AI phrase density, inverse lexical diversity, and sentence uniformity across all captions
- Computes `lexical_diversity` (Type-Token Ratio) across the full post corpus

#### 3.5.2 Agenda Detection
- Scans for 22 spam keywords (e.g., "earn money", "link in bio", "giveaway") and 13 agenda keywords (e.g., "vote", "wake up", "expose")
- Measures **hashtag repetition**: ratio of repeated to unique hashtags across all posts
- Measures **topic concentration**: fraction of total tokens covered by the top-5 most frequent content words
- Final `agenda_score` = weighted combination (0.35 spam + 0.25 agenda keywords + 0.20 hashtag + 0.20 topic)

#### 3.5.3 Temporal Behavior Analysis
- Parses post timestamps (ISO strings, UNIX integers, or datetime objects)
- `posts_per_day` = post count ÷ span (days)
- `avg_time_gap_hours` = mean gap between consecutive posts
- `time_variance_score` = 1 − (std of posting hour / 12); score of 1.0 means all posts at the same clock time (bot-like)

#### 3.5.4 Engagement Ratio
- `engagement_ratio` = mean(likes per post) / follower\_count
- Genuine accounts typically show 1–5% engagement; purchased-follower accounts show < 0.1%

#### 3.5.5 Video / Audio Transcription
For each video post with a `media_url`:
1. **FFmpeg** (`subprocess`) downloads and extracts 16kHz mono WAV audio
2. **OpenAI Whisper** (`tiny` model, CPU-only, ~150 MB, fully local) transcribes audio to text
3. Transcription is appended to the post's caption text for all subsequent NLP analysis
4. Gracefully skips if FFmpeg is not installed or Whisper is unavailable

#### 3.5.6 Cross-Content Consistency
- `content_similarity_score` = TF-IDF cosine similarity between bio text and all post captions combined
- Very high (> 0.92) → suspiciously scripted; very low (< 0.03) with non-empty bio → identity mismatch

### 3.6 Rule-Based Probability Adjustment (No Retraining)
After all layers complete, `apply_behavior_adjustments()` modulates the ML model's `fake_probability`:

| Rule | Threshold | Adjustment |
|---|---|---|
| High posting freq + low time variance | ppd > 10 AND tvar > 0.75 | +0.20 |
| Moderate posting + regular schedule | ppd > 5 AND tvar > 0.55 | +0.12 |
| High duplicate + high AI text | dup > 0.70 AND ai > 0.50 | +0.22 |
| Moderate duplicate OR AI | dup > 0.45 OR ai > 0.45 | +0.10 |
| Image reuse + near-zero engagement | reuse AND eng < 0.01 | +0.25 |
| Image reuse alone | reuse | +0.12 |
| Strong agenda + uniform content | agenda > 0.60 AND sim > 0.70 | +0.15 |
| Mild agenda | agenda > 0.35 | +0.06 |
| Very low face ratio (≥5 posts) | face < 0.15 | +0.08 |
| Near-zero engagement (≥3 posts) | eng < 0.003 | +0.10 |

Final probability is hard-capped at 0.99. All adjustments are cumulative; the final threshold is ≥ 0.50 → FAKE.

---

## 4. API Architecture

The system exposes five REST endpoints via Django REST Framework:

| Endpoint | Method | Description |
|---|---|---|
| `/api/predict-profile/` | POST | Structural ML prediction only |
| `/api/detect-ai-text/` | POST | NLP AI text detection only |
| `/api/analyze/` | POST | ML + NLP + Image (manual data) |
| `/api/analyze-url/` | POST | Scrape Instagram + full analysis |
| `/api/analyze-posts/` | POST | Full 4-layer analysis with posts |

The new `/api/analyze-posts/` endpoint accepts:
```json
{
  "profile": { "bio": "...", "num_followers": 1200, ... },
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
```

---

## 5. Experimental Results

### 5.1 Core Model Metrics (Random Forest)
Evaluated on an 80/20 stratified split of 576 labeled profiles:
- **Accuracy:** 90.52%
- **ROC-AUC Score:** 0.9819
- **Precision (Fake):** 89% | **Recall (Fake):** 93%

### 5.2 Feature Importance (Gini Impurity)
1. **Followers Count (23.4%)** — Absolute follower count is hardest to fake organically
2. **Post Count (18.0%)** — Genuine accounts accumulate posts over years; bots deploy rapidly
3. **Followers/Following Ratio (9.9%)** — Critical engineered feature confirmed by Gini score
4. **Profile Picture Presence (9.8%)** — Immediate negative signal when absent

### 5.3 False Positive Analysis
The confusion matrix showed only 7 False Positives (real profiles marked fake) out of 116 test samples. Most false positives were new users with zero posts and zero bio — a classic "cold start" problem addressable by the BCM which can provide more nuanced signals when post data is available.

---

## 6. Technology Stack

| Component | Technology | Purpose |
|---|---|---|
| Backend | Django 6.x + DRF | REST API server |
| ML Classifier | scikit-learn Random Forest | Structural profile scoring |
| NLP Detection | Python (re, collections, math) | AI text heuristics |
| Image Analysis | OpenCV + Pillow + ImageHash | Face detection + pHash reuse |
| Text Similarity | scikit-learn TfidfVectorizer | Cross-content consistency |
| Video/Audio | FFmpeg + OpenAI Whisper (tiny) | Audio extraction + transcription |
| Frontend | Vanilla HTML/CSS/JS | Interactive UI |
| Data | SQLite + pandas | Storage + preprocessing |

---

## 7. Limitations & Future Work

### 7.1 Current Limitations
- **Instagram scraping:** Instagram's anti-scraping measures limit public API access; post data must be submitted manually via the `/api/analyze-posts/` endpoint
- **Whisper accuracy:** The `tiny` Whisper model trades accuracy for speed; upgrade to `base` or `small` for better transcription quality
- **Label drift:** The 576-profile training dataset is static; bot tactics evolve and the model may need periodic retraining on new data

### 7.2 Future Enhancement Areas

#### A. Deepfake/GAN Detection
**Current:** Face presence is detected via Haar cascade (binary). **Enhancement:** Replace with a CNN-based GAN artifact detector (e.g., EfficientNet fine-tuned on FaceForensics++ dataset) to distinguish AI-generated profile photos from real photographs.

#### B. Transformer-Based NLP
**Current:** Heuristic phrase counting. **Enhancement:** Fine-tune DistilBERT or DeBERTa specifically on human-vs-AI social media bios, introducing semantic understanding beyond word counting.

#### C. Graph Neural Network Follower Analysis
**Current:** Follower/following ratio is a scalar feature. **Enhancement:** Construct a follower subgraph using scraped follower data; apply a Graph Convolutional Network (GCN) to identify bot "rings" that exclusively cross-follow each other.

#### D. Real-Time Monitoring & Time-Series
**Current:** Static snapshot analysis. **Enhancement:** Track follower growth velocity over time; a sudden spike of 10,000 followers gained in 2 hours then flatlining is a guaranteed hallmark of purchased bot traffic.

#### E. Multimodal Deepfake Alignment
**Current:** Video audio is transcribed and analyzed as text. **Enhancement:** Cross-reference audio speaker identity (via voice embedding) with post-image face identity to detect accounts presenting a fake persona with voice/appearance inconsistencies.

---

## 8. Conclusion

This multi-signal AI system successfully bridges supervised tabular machine learning, heuristic NLP, computer vision, and behavioral content analysis to flag fraudulent social media operators with high accuracy and explainability. The four-layer architecture ensures that sophisticated fake accounts—which may pass any single detection layer—are caught by at least one of: (1) structural ML anomalies, (2) AI-generated bio text, (3) duplicate or faceless profile images, or (4) scripted, agenda-driven, or temporally robotic posting behavior including detection of spoken content in videos. Operating at a core accuracy exceeding 90%, with behavior adjustments pushing the combined system significantly higher on multi-signal cases, the transparent architecture translates abstract probability into actionable, human-readable explanations at every layer.
