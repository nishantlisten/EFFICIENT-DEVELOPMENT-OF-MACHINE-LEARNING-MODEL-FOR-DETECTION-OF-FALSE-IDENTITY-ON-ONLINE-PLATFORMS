# AI Fake Profile Detection System — Setup Guide

## Prerequisites

- **Python 3.10+** (Download from [python.org](https://www.python.org/downloads/))
- **pip** (comes with Python)
- **Git** (optional, for cloning)

---

## Step 1: Get the Project

```bash
# Option A: If you have the zip file
unzip fake_profile.zip
cd fake_profile

# Option B: If cloning from a repository
git clone <repository-url>
cd fake_profile
```

---

## Step 2: Create Virtual Environment (Recommended)

```bash
# Create virtual environment
python3 -m venv venv

# Activate it
# On macOS/Linux:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

---

## Step 3: Install Dependencies

```bash
pip install scikit-learn pandas numpy matplotlib seaborn joblib
pip install django djangorestframework django-cors-headers
pip install instaloader
```

Or install from requirements file:

```bash
pip install -r requirements.txt
```

---

## Step 4: Train the ML Model

```bash
python3 ml_model/train_model.py
```

**Expected output:**
```
  ✅ Accuracy:  0.9052  (90.52%)
  📈 ROC-AUC:   0.9819
  💾 Model saved to: ml_model/fake_profile_model.pkl
  💾 Scaler saved to: ml_model/scaler.pkl
```

> **Note:** If the model files (`fake_profile_model.pkl`, `scaler.pkl`, `feature_columns.pkl`) already exist in `ml_model/`, you can skip this step.

---

## Step 5: Run Database Migrations

```bash
cd backend
python3 manage.py migrate
```

---

## Step 6: Start the Server

```bash
python3 manage.py runserver 8000
```

**Expected output:**
```
✅ ML model loaded successfully!
Starting development server at http://127.0.0.1:8000/
```

---

## Step 7: Open in Browser

Open your browser and navigate to:

```
http://127.0.0.1:8000/
```

---

## Usage

### Option 1: Analyze by URL (Default)
1. Paste an Instagram profile URL (e.g., `https://instagram.com/username`)
2. Click **"Fetch & Analyze"**
3. The system auto-fetches profile data and shows the verdict

### Option 2: Manual Entry
1. Click the **"Manual Entry"** tab
2. Enter profile details (username, followers, following, posts, bio, etc.)
3. Click **"Analyze Profile"**
4. View the FAKE/REAL verdict with confidence score and explanations

---

## API Endpoints (for developers)

| Endpoint | Method | Description |
|---|---|---|
| `/api/analyze-url/` | POST | Send `{"url": "https://instagram.com/username"}` |
| `/api/analyze/` | POST | Send profile data JSON for combined analysis |
| `/api/predict-profile/` | POST | Profile ML prediction only |
| `/api/detect-ai-text/` | POST | Send `{"text": "..."}` for AI text detection |

**Example API test with curl:**
```bash
curl -X POST http://127.0.0.1:8000/api/analyze/ \
  -H "Content-Type: application/json" \
  -d '{
    "username": "bot_user_123",
    "fullname": "Bot",
    "bio": "Passionate about leveraging technology",
    "num_followers": 5,
    "num_following": 800,
    "num_posts": 0,
    "profile_pic": false,
    "external_url": false,
    "private": false
  }'
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError: No module named 'django'` | Run `pip install django djangorestframework django-cors-headers` |
| `ModuleNotFoundError: No module named 'sklearn'` | Run `pip install scikit-learn` |
| `ML model not loaded` | Run `python3 ml_model/train_model.py` first |
| `InconsistentVersionWarning` (sklearn version) | Re-run `python3 ml_model/train_model.py` to retrain with your version |
| Instagram scraping fails / rate-limited | Use the "Manual Entry" tab instead |
| CSS/JS not loading (unstyled page) | Make sure you're running from the `backend/` directory |
| Port 8000 already in use | Use `python3 manage.py runserver 8001` and open `localhost:8001` |

---

## Project Structure

```
fake_profile/
├── ml_model/
│   ├── train_model.py          # Training script
│   ├── ai_text_detector.py     # AI text detection
│   ├── profile_scraper.py      # Instagram URL scraper
│   ├── fake_profile_model.pkl  # Trained model
│   ├── scaler.pkl              # Feature scaler
│   └── feature_columns.pkl     # Feature names
├── backend/
│   ├── manage.py               # Django entry point
│   ├── backend/settings.py     # Configuration
│   └── api/views.py            # API endpoints
├── frontend/
│   ├── index.html              # Web UI
│   ├── style.css               # Styling
│   └── script.js               # Frontend logic
├── instagram-fake-spammer-genuine-accounts/
│   └── train.csv               # Training dataset
├── PROJECT_REPORT.md           # Full project report
├── README.md                   # This file
└── requirements.txt            # Python dependencies
```
