/**
 * ══════════════════════════════════════════════════════════════
 *  AI Fake Profile Detector — Frontend Logic
 * ══════════════════════════════════════════════════════════════
 * Handles all tabs: URL analysis, Manual entry, Post Analysis
 * Includes behavior signal rendering and post row builder.
 */

// ── Configuration ─────────────────────────────────────────────
const isLocal = window.location.hostname === 'localhost' ||
    window.location.hostname === '127.0.0.1' ||
    window.location.origin === 'file://' ||
    window.location.origin === 'null';
const API_BASE = isLocal ? 'http://127.0.0.1:8000' : window.location.origin;
const ENDPOINTS = {
    analyze: `${API_BASE}/api/analyze/`,
    analyzeUrl: `${API_BASE}/api/analyze-url/`,
    analyzePosts: `${API_BASE}/api/analyze-posts/`,
    predictProfile: `${API_BASE}/api/predict-profile/`,
    detectAiText: `${API_BASE}/api/detect-ai-text/`,
};

// ── DOM Elements ──────────────────────────────────────────────
const urlForm = document.getElementById('url-form');
const manualForm = document.getElementById('analysis-form');
const postsForm = document.getElementById('posts-form');
const urlSection = document.getElementById('url-section');
const manualSection = document.getElementById('manual-section');
const postsSection = document.getElementById('posts-section');
const scrapedSection = document.getElementById('scraped-section');
const scrapedGrid = document.getElementById('scraped-grid');
const resultsSection = document.getElementById('results-section');
const behaviorCard = document.getElementById('behavior-card');

// Verdict elements
const verdictBadge = document.getElementById('verdict-badge');
const verdictIcon = document.getElementById('verdict-icon');
const verdictValue = document.getElementById('verdict-value');
const confidenceValue = document.getElementById('confidence-value');
const confidenceFill = document.getElementById('confidence-fill');

// Analysis elements
const profileBadge = document.getElementById('profile-badge');
const profileProb = document.getElementById('profile-prob');
const profileFill = document.getElementById('profile-fill');
const profileReasons = document.getElementById('profile-reasons');
const aiBadge = document.getElementById('ai-badge');
const aiProb = document.getElementById('ai-prob');
const aiFill = document.getElementById('ai-fill');
const aiReasons = document.getElementById('ai-reasons');
const scoresGrid = document.getElementById('scores-grid');
const allReasons = document.getElementById('all-reasons');
const behaviorBadge = document.getElementById('behavior-badge');
const behaviorSignalsGrid = document.getElementById('behavior-signals-grid');
const behaviorReasons = document.getElementById('behavior-reasons');

// ── Post Row State ────────────────────────────────────────────
let postRows = [];
let postRowCounter = 0;

// ── Tab Switching ─────────────────────────────────────────────
function switchTab(mode) {
    const tabUrl = document.getElementById('tab-url');
    const tabManual = document.getElementById('tab-manual');
    const tabPosts = document.getElementById('tab-posts');

    [tabUrl, tabManual, tabPosts].forEach(t => t.classList.remove('active'));
    [urlSection, manualSection, postsSection].forEach(s => s.style.display = 'none');

    if (mode === 'url') {
        tabUrl.classList.add('active');
        urlSection.style.display = 'block';
    } else if (mode === 'manual') {
        tabManual.classList.add('active');
        manualSection.style.display = 'block';
    } else {
        tabPosts.classList.add('active');
        postsSection.style.display = 'block';
    }

    resultsSection.style.display = 'none';
    scrapedSection.style.display = 'none';
    behaviorCard.style.display = 'none';
}

// ── Post Row Builder ──────────────────────────────────────────
function addPostRow() {
    const id = ++postRowCounter;
    const now = new Date();
    const isoNow = now.toISOString().slice(0, 16); // YYYY-MM-DDTHH:MM

    const row = document.createElement('div');
    row.className = 'post-row';
    row.id = `post-row-${id}`;
    row.innerHTML = `
        <div class="post-row-header">
            <span class="post-row-num">Post #${id}</span>
            <button type="button" class="btn-remove-post" onclick="removePostRow(${id})">
                <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>
            </button>
        </div>
        <div class="post-row-grid">
            <div class="form-group">
                <label>Timestamp</label>
                <input type="datetime-local" id="post-ts-${id}" value="${isoNow}">
            </div>
            <div class="form-group">
                <label>Media Type</label>
                <select id="post-type-${id}" class="post-select">
                    <option value="image">🖼️ Image</option>
                    <option value="video">🎬 Video</option>
                </select>
            </div>
            <div class="form-group">
                <label>Likes</label>
                <input type="number" id="post-likes-${id}" placeholder="0" min="0" value="0">
            </div>
        </div>
        <div class="form-group">
            <label>Caption / Text</label>
            <textarea id="post-caption-${id}" rows="2" placeholder="Post caption or description..."></textarea>
        </div>
        <div class="form-group">
            <label>Media URL <span class="optional-tag">(optional — for image face & video audio analysis)</span></label>
            <input type="url" id="post-url-${id}" placeholder="https://...">
        </div>
    `;

    document.getElementById('posts-list').appendChild(row);
    postRows.push(id);
    updatePostCount();

    // Animate in
    requestAnimationFrame(() => row.classList.add('visible'));
}

function removePostRow(id) {
    const row = document.getElementById(`post-row-${id}`);
    if (row) {
        row.classList.add('removing');
        setTimeout(() => {
            row.remove();
            postRows = postRows.filter(r => r !== id);
            updatePostCount();
        }, 250);
    }
}

function updatePostCount() {
    document.getElementById('post-count-display').textContent = postRows.length;
}

function collectPosts() {
    return postRows.map(id => ({
        timestamp: document.getElementById(`post-ts-${id}`)?.value || '',
        caption: document.getElementById(`post-caption-${id}`)?.value?.trim() || '',
        likes: parseInt(document.getElementById(`post-likes-${id}`)?.value) || 0,
        media_type: document.getElementById(`post-type-${id}`)?.value || 'image',
        media_url: document.getElementById(`post-url-${id}`)?.value?.trim() || '',
    }));
}

// ── URL Form Submission ───────────────────────────────────────
urlForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    const url = document.getElementById('profile-url').value.trim();
    if (!url) return;

    const btn = document.getElementById('btn-url-analyze');
    btn.classList.add('loading');
    resultsSection.style.display = 'none';
    scrapedSection.style.display = 'none';

    try {
        const response = await fetch(ENDPOINTS.analyzeUrl, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ url }),
        });

        const result = await response.json();

        if (!response.ok) {
            throw new Error(result.error || 'Failed to analyze profile');
        }

        if (result.scraped_profile) {
            displayScrapedProfile(result.scraped_profile);
        }

        displayResults(result, false);

    } catch (error) {
        console.error('URL analysis failed:', error);
        showError(error.message);
    } finally {
        btn.classList.remove('loading');
    }
});

// ── Manual Form Submission ────────────────────────────────────
manualForm.addEventListener('submit', async (e) => {
    e.preventDefault();

    const data = {
        username: document.getElementById('username').value.trim(),
        fullname: document.getElementById('fullname').value.trim(),
        bio: document.getElementById('bio').value.trim(),
        num_followers: parseInt(document.getElementById('num_followers').value) || 0,
        num_following: parseInt(document.getElementById('num_following').value) || 0,
        num_posts: parseInt(document.getElementById('num_posts').value) || 0,
        profile_pic: document.getElementById('profile_pic').checked,
        external_url: document.getElementById('external_url').checked,
        private: document.getElementById('private').checked,
        desc_length: document.getElementById('bio').value.trim().length,
    };

    const btn = document.getElementById('btn-analyze');
    btn.classList.add('loading');
    resultsSection.style.display = 'none';
    scrapedSection.style.display = 'none';

    try {
        const response = await fetch(ENDPOINTS.analyze, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data),
        });

        if (!response.ok) {
            const err = await response.json();
            throw new Error(err.error || 'API request failed');
        }

        const result = await response.json();
        displayResults(result, false);

    } catch (error) {
        console.error('Analysis failed:', error);
        showError(error.message);
    } finally {
        btn.classList.remove('loading');
    }
});

// ── Posts Form Submission ─────────────────────────────────────
postsForm.addEventListener('submit', async (e) => {
    e.preventDefault();

    const profile = {
        bio: document.getElementById('p-bio').value.trim(),
        num_followers: parseInt(document.getElementById('p-followers').value) || 0,
        num_following: parseInt(document.getElementById('p-following').value) || 0,
        num_posts: parseInt(document.getElementById('p-num-posts').value) || 0,
        profile_pic: true,
    };

    const posts = collectPosts();

    if (posts.length === 0) {
        showError('Please add at least one post to analyze. Click "Add Post" button.');
        return;
    }

    const btn = document.getElementById('btn-posts-analyze');
    btn.classList.add('loading');
    resultsSection.style.display = 'none';

    try {
        const response = await fetch(ENDPOINTS.analyzePosts, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ profile, posts }),
        });

        if (!response.ok) {
            const err = await response.json();
            throw new Error(err.error || 'Post analysis failed');
        }

        const result = await response.json();
        displayResults(result, true);

    } catch (error) {
        console.error('Post analysis failed:', error);
        showError(error.message);
    } finally {
        btn.classList.remove('loading');
    }
});

// ── Display Scraped Profile ───────────────────────────────────
function displayScrapedProfile(profile) {
    scrapedGrid.innerHTML = '';

    const fields = [
        { label: 'Username', value: `@${profile.username}`, icon: '👤' },
        { label: 'Full Name', value: profile.fullname || '—', icon: '📛' },
        { label: 'Followers', value: formatNumber(profile.num_followers), icon: '👥' },
        { label: 'Following', value: formatNumber(profile.num_following), icon: '➡️' },
        { label: 'Posts', value: formatNumber(profile.num_posts), icon: '📷' },
        { label: 'Profile Pic', value: profile.profile_pic ? 'Yes' : 'No', icon: '🖼️' },
        { label: 'External URL', value: profile.external_url ? 'Yes' : 'No', icon: '🔗' },
        { label: 'Private', value: profile.private ? 'Yes' : 'No', icon: '🔒' },
        { label: 'Verified', value: profile.is_verified ? '✅ Yes' : 'No', icon: '✓' },
    ];

    fields.forEach(f => {
        const item = document.createElement('div');
        item.className = 'scraped-item';
        item.innerHTML = `
            <span class="scraped-icon">${f.icon}</span>
            <div class="scraped-info">
                <span class="scraped-label">${f.label}</span>
                <span class="scraped-value">${f.value}</span>
            </div>
        `;
        scrapedGrid.appendChild(item);
    });

    if (profile.bio) {
        const bioItem = document.createElement('div');
        bioItem.className = 'scraped-item scraped-bio';
        bioItem.innerHTML = `
            <span class="scraped-icon">📝</span>
            <div class="scraped-info">
                <span class="scraped-label">Bio</span>
                <span class="scraped-value bio-text">${profile.bio}</span>
            </div>
        `;
        scrapedGrid.appendChild(bioItem);
    }

    scrapedSection.style.display = 'block';
}

// ── Display Results ───────────────────────────────────────────
function displayResults(result, showBehavior = false) {
    const isFake = result.is_fake;
    const confidence = result.confidence;

    // Final Verdict
    verdictBadge.className = `verdict-badge ${isFake ? 'fake' : 'real'}`;
    verdictIcon.textContent = isFake ? '🚨' : '✅';
    verdictValue.textContent = result.final_verdict;
    confidenceValue.textContent = `${confidence}%`;
    confidenceFill.className = `confidence-fill ${isFake ? 'fake' : 'real'}`;
    setTimeout(() => { confidenceFill.style.width = `${confidence}%`; }, 100);

    // Profile Analysis
    const profile = result.profile_analysis;
    const profileIsFake = profile.prediction === 'FAKE';
    profileBadge.textContent = profile.prediction;
    profileBadge.className = `analysis-badge ${profileIsFake ? 'fake' : 'real'}`;
    profileProb.textContent = `${profile.fake_probability}%`;
    setTimeout(() => { profileFill.style.width = `${profile.fake_probability}%`; }, 200);

    profileReasons.innerHTML = '';
    (profile.reasons || []).forEach((reason, i) => {
        profileReasons.appendChild(createReasonItem(reason, i * 50));
    });

    // AI Text Analysis
    const ai = result.ai_text_analysis;
    const isAi = ai.is_ai_generated;
    aiBadge.textContent = isAi ? 'AI DETECTED' : 'HUMAN';
    aiBadge.className = `analysis-badge ${isAi ? 'ai' : 'human'}`;
    aiProb.textContent = `${ai.confidence}%`;
    setTimeout(() => { aiFill.style.width = `${ai.confidence}%`; }, 300);

    aiReasons.innerHTML = '';
    (ai.reasons || []).forEach((reason, i) => {
        aiReasons.appendChild(createReasonItem(reason, i * 50, 'ai-warning'));
    });

    // Score breakdown
    scoresGrid.innerHTML = '';
    if (ai.scores && Object.keys(ai.scores).length > 0) {
        Object.entries(ai.scores).forEach(([key, value]) => {
            const item = document.createElement('div');
            item.className = 'score-item';
            item.innerHTML = `
                <span class="score-name">${formatScoreName(key)}</span>
                <span class="score-val">${(value * 100).toFixed(1)}%</span>
            `;
            scoresGrid.appendChild(item);
        });
    }

    // Behavior Analysis (only for posts tab)
    if (showBehavior && result.behavior_analysis) {
        displayBehaviorAnalysis(result.behavior_analysis);
        behaviorCard.style.display = 'block';
    } else {
        behaviorCard.style.display = 'none';
    }

    // All Combined Reasons
    allReasons.innerHTML = '';
    (result.reasons || []).forEach((reason, i) => {
        const type = reason.startsWith('✅') ? 'success' :
                     reason.startsWith('❌') ? 'warning' :
                     reason.startsWith('⚠️') ? 'ai-warning' :
                     reason.startsWith('🚨') ? 'warning' :
                     reason.startsWith('🤖') ? 'bot-warning' : '';
        allReasons.appendChild(createReasonItem(reason, i * 60, type));
    });

    resultsSection.style.display = 'block';
    resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// ── Behavior Analysis Display ─────────────────────────────────
function displayBehaviorAnalysis(behavior) {
    const hasSuspiciousSignal = (
        behavior.posts_per_day > 5 ||
        behavior.time_variance_score > 0.55 ||
        behavior.duplicate_text_ratio > 0.45 ||
        behavior.ai_text_score > 0.30 ||
        behavior.agenda_score > 0.35 ||
        behavior.image_reuse_detected
    );

    behaviorBadge.textContent = hasSuspiciousSignal ? 'SUSPICIOUS' : 'NORMAL';
    behaviorBadge.className = `analysis-badge ${hasSuspiciousSignal ? 'fake' : 'real'}`;

    // Signal meter cards
    behaviorSignalsGrid.innerHTML = '';
    const signals = [
        {
            label: 'Posts / Day',
            value: behavior.posts_per_day.toFixed(2),
            raw: Math.min(behavior.posts_per_day / 15, 1),
            danger: behavior.posts_per_day > 5,
            icon: '📅',
            unit: ''
        },
        {
            label: 'Timing Regularity',
            value: (behavior.time_variance_score * 100).toFixed(0) + '%',
            raw: behavior.time_variance_score,
            danger: behavior.time_variance_score > 0.55,
            icon: '⏱️',
            unit: ''
        },
        {
            label: 'Caption Similarity',
            value: (behavior.duplicate_text_ratio * 100).toFixed(0) + '%',
            raw: behavior.duplicate_text_ratio,
            danger: behavior.duplicate_text_ratio > 0.45,
            icon: '📋',
            unit: ''
        },
        {
            label: 'AI Text Score',
            value: (behavior.ai_text_score * 100).toFixed(0) + '%',
            raw: behavior.ai_text_score,
            danger: behavior.ai_text_score > 0.30,
            icon: '🤖',
            unit: ''
        },
        {
            label: 'Agenda Score',
            value: (behavior.agenda_score * 100).toFixed(0) + '%',
            raw: behavior.agenda_score,
            danger: behavior.agenda_score > 0.35,
            icon: '📣',
            unit: ''
        },
        {
            label: 'Engagement Ratio',
            value: (behavior.engagement_ratio * 100).toFixed(2) + '%',
            raw: Math.min(behavior.engagement_ratio * 20, 1),
            danger: behavior.engagement_ratio < 0.005 && behavior.posts_analyzed > 0,
            icon: '💬',
            unit: '',
            invert: true
        },
        {
            label: 'Face Presence',
            value: (behavior.face_presence_ratio * 100).toFixed(0) + '%',
            raw: behavior.face_presence_ratio,
            danger: behavior.face_presence_ratio < 0.2 && behavior.posts_analyzed >= 3,
            icon: '👤',
            unit: '',
            invert: true
        },
        {
            label: 'Lexical Diversity',
            value: (behavior.lexical_diversity * 100).toFixed(0) + '%',
            raw: behavior.lexical_diversity,
            danger: behavior.lexical_diversity < 0.5,
            icon: '📖',
            unit: '',
            invert: true
        },
    ];

    signals.forEach((sig, idx) => {
        const card = document.createElement('div');
        card.className = `signal-card ${sig.danger ? 'signal-danger' : 'signal-ok'}`;
        card.style.animationDelay = `${idx * 60}ms`;
        card.innerHTML = `
            <div class="signal-icon">${sig.icon}</div>
            <div class="signal-body">
                <span class="signal-label">${sig.label}</span>
                <span class="signal-value ${sig.danger ? 'danger-val' : 'ok-val'}">${sig.value}</span>
            </div>
            <div class="signal-bar-wrap">
                <div class="signal-bar-fill ${sig.danger ? 'danger-fill' : 'ok-fill'}" style="width: ${(sig.raw * 100).toFixed(0)}%"></div>
            </div>
        `;
        behaviorSignalsGrid.appendChild(card);
    });

    // Extra info row
    const extras = document.createElement('div');
    extras.className = 'behavior-extras';
    extras.innerHTML = `
        <span class="behavior-tag">Posts analyzed: <strong>${behavior.posts_analyzed}</strong></span>
        <span class="behavior-tag ${behavior.image_reuse_detected ? 'tag-danger' : ''}">Image reuse: <strong>${behavior.image_reuse_detected ? '⚠️ Detected' : '✅ None'}</strong></span>
        <span class="behavior-tag">Videos transcribed: <strong>${behavior.video_transcripts_count}</strong></span>
        <span class="behavior-tag ${behavior.ffmpeg_available ? '' : 'tag-muted'}">FFmpeg: <strong>${behavior.ffmpeg_available ? '✅' : '⚠️ Not found'}</strong></span>
        <span class="behavior-tag ${behavior.whisper_available ? '' : 'tag-muted'}">Whisper STT: <strong>${behavior.whisper_available ? '✅' : '⚠️ Unavailable'}</strong></span>
    `;
    behaviorSignalsGrid.appendChild(extras);

    // Behavior-specific reasons
    behaviorReasons.innerHTML = '';
    (behavior.reasons || []).forEach((reason, i) => {
        behaviorReasons.appendChild(createReasonItem(reason, i * 50));
    });
}

// ── Helpers ───────────────────────────────────────────────────
function createReasonItem(text, delay = 0, type = '') {
    const item = document.createElement('div');
    item.className = `reason-item ${type || getReasonType(text)}`;
    item.textContent = text;
    item.style.animationDelay = `${delay}ms`;
    return item;
}

function getReasonType(text) {
    if (text.includes('❌') || text.includes('🚨')) return 'warning';
    if (text.includes('✅')) return 'success';
    if (text.includes('⚠️')) return 'ai-warning';
    if (text.includes('🤖')) return 'bot-warning';
    if (text.includes('ℹ️')) return 'info';
    return '';
}

function formatScoreName(key) {
    return key.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
}

function formatNumber(n) {
    if (n >= 1000000) return `${(n / 1000000).toFixed(1)}M`;
    if (n >= 1000) return `${(n / 1000).toFixed(1)}K`;
    return String(n);
}

function showError(message) {
    resultsSection.style.display = 'block';
    behaviorCard.style.display = 'none';
    verdictBadge.className = 'verdict-badge fake';
    verdictIcon.textContent = '⚠️';
    verdictValue.textContent = 'ERROR';
    confidenceValue.textContent = '—';
    confidenceFill.style.width = '0%';
    profileReasons.innerHTML = '';
    aiReasons.innerHTML = '';
    scoresGrid.innerHTML = '';
    allReasons.innerHTML = '';
    const item = document.createElement('div');
    item.className = 'reason-item warning';
    item.textContent = `${message}`;
    allReasons.appendChild(item);
    resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// ── Input Animations ──────────────────────────────────────────
document.querySelectorAll('.form-group input, .form-group textarea, .url-input-wrapper input').forEach(el => {
    el.addEventListener('focus', () => {
        const parent = el.closest('.form-group, .url-input-wrapper');
        if (parent) parent.style.transform = 'translateY(-2px)';
    });
    el.addEventListener('blur', () => {
        const parent = el.closest('.form-group, .url-input-wrapper');
        if (parent) parent.style.transform = 'translateY(0)';
    });
});

// Add first post row by default when switching to posts tab
document.getElementById('tab-posts').addEventListener('click', () => {
    if (postRows.length === 0) {
        setTimeout(addPostRow, 150);
    }
});
