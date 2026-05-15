import streamlit as st
import pickle
import re
import nltk
import plotly.graph_objects as go
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

nltk.download('stopwords', quiet=True)

# ── PAGE CONFIG ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SentiScope · Review Intelligence",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── CUSTOM CSS ─────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&display=swap');

*, *::before, *::after { box-sizing: border-box; }

html, body, .stApp {
    background-color: #060910 !important;
    font-family: 'DM Sans', sans-serif !important;
    color: #e2e8f0 !important;
}

#MainMenu, footer, header { visibility: hidden; }

.block-container {
    padding: 2rem 3rem !important;
    max-width: 1200px !important;
}

/* ── ANIMATED BACKGROUND ── */
.stApp::before {
    content: '';
    position: fixed;
    top: 0; left: 0; right: 0; bottom: 0;
    background:
        radial-gradient(ellipse at 20% 20%, rgba(59,130,246,0.08) 0%, transparent 50%),
        radial-gradient(ellipse at 80% 80%, rgba(139,92,246,0.08) 0%, transparent 50%),
        radial-gradient(ellipse at 60% 10%, rgba(16,185,129,0.05) 0%, transparent 40%);
    pointer-events: none;
    z-index: 0;
}

/* ── HEADER ── */
.sentiscope-header {
    display: flex;
    align-items: center;
    gap: 16px;
    margin-bottom: 8px;
}
.sentiscope-logo {
    width: 52px;
    height: 52px;
    background: linear-gradient(135deg, #3b82f6, #8b5cf6);
    border-radius: 14px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 26px;
    box-shadow: 0 0 30px rgba(59,130,246,0.4);
}
.sentiscope-title {
    font-size: 36px;
    font-weight: 700;
    background: linear-gradient(135deg, #60a5fa, #a78bfa, #34d399);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0;
    letter-spacing: -1px;
}
.sentiscope-subtitle {
    font-size: 14px;
    color: #475569;
    margin: 0 0 2rem 0;
    font-weight: 400;
    letter-spacing: 0.03em;
}
.version-chip {
    display: inline-block;
    font-size: 11px;
    font-family: 'DM Mono', monospace;
    background: linear-gradient(135deg, rgba(59,130,246,0.15), rgba(139,92,246,0.15));
    color: #7dd3fc;
    border: 1px solid rgba(59,130,246,0.3);
    border-radius: 20px;
    padding: 3px 12px;
    margin-left: 10px;
    vertical-align: middle;
}

/* ── DIVIDER ── */
.custom-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, #1e3a5f, #312e81, #1e3a5f, transparent);
    margin: 1.8rem 0;
    border: none;
}

/* ── STATS ROW ── */
.stats-row {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 14px;
    margin-bottom: 1.8rem;
}
.stat-card {
    background: linear-gradient(135deg, #0f172a, #111827);
    border: 1px solid #1e293b;
    border-radius: 16px;
    padding: 18px 20px;
    text-align: center;
    position: relative;
    overflow: hidden;
    transition: transform 0.2s, box-shadow 0.2s;
}
.stat-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, #3b82f6, #8b5cf6);
}
.stat-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 30px rgba(59,130,246,0.15);
}
.stat-value {
    font-size: 28px;
    font-weight: 700;
    background: linear-gradient(135deg, #60a5fa, #a78bfa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    font-family: 'DM Mono', monospace;
    letter-spacing: -1px;
}
.stat-label {
    font-size: 12px;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-top: 4px;
    font-weight: 500;
}

/* ── INPUT LABEL ── */
.input-label {
    font-size: 15px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #7dd3fc;
    margin-bottom: 12px;
    display: flex;
    align-items: center;
    gap: 8px;
}

/* ── TEXTAREA ── */
.stTextArea textarea {
    background: #0d1424 !important;
    border: 1px solid #1e3a5f !important;
    border-radius: 12px !important;
    color: #e2e8f0 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 15px !important;
    padding: 16px !important;
    transition: border-color 0.2s, box-shadow 0.2s !important;
    line-height: 1.6 !important;
}
.stTextArea textarea:focus {
    border-color: #3b82f6 !important;
    box-shadow: 0 0 0 3px rgba(59,130,246,0.15), 0 0 20px rgba(59,130,246,0.1) !important;
}
.stTextArea textarea::placeholder { color: #2d3f55 !important; }

/* ── BUTTON ── */
.stButton > button {
    background: linear-gradient(135deg, #2563eb, #7c3aed, #0ea5e9) !important;
    background-size: 200% 200% !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 15px !important;
    font-weight: 600 !important;
    padding: 0.75rem 1.5rem !important;
    width: 100% !important;
    letter-spacing: 0.03em !important;
    box-shadow: 0 4px 20px rgba(37,99,235,0.3) !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 30px rgba(37,99,235,0.5) !important;
    opacity: 0.95 !important;
}

/* ── RESULT CARDS ── */
.result-card {
    background: linear-gradient(135deg, #0f172a, #111827);
    border: 1px solid #1e293b;
    border-radius: 18px;
    padding: 1.6rem 1.8rem;
    height: 100%;
    position: relative;
    overflow: hidden;
    transition: transform 0.2s, box-shadow 0.2s;
}
.result-card:hover { transform: translateY(-2px); }
.result-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
}
.result-card.positive {
    box-shadow: 0 0 40px rgba(16,185,129,0.08);
}
.result-card.positive::before {
    background: linear-gradient(90deg, #10b981, #34d399, #6ee7b7);
}
.result-card.negative {
    box-shadow: 0 0 40px rgba(239,68,68,0.08);
}
.result-card.negative::before {
    background: linear-gradient(90deg, #ef4444, #f87171, #fca5a5);
}
.result-card.neutral {
    box-shadow: 0 0 40px rgba(245,158,11,0.08);
}
.result-card.neutral::before {
    background: linear-gradient(90deg, #f59e0b, #fbbf24, #fde68a);
}
.result-card-label {
    font-size: 13px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #94a3b8;
    margin-bottom: 14px;
}
.result-sentiment {
    font-size: 36px;
    font-weight: 700;
    margin: 6px 0;
    letter-spacing: -1px;
}
.sentiment-positive {
    background: linear-gradient(135deg, #10b981, #34d399);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.sentiment-negative {
    background: linear-gradient(135deg, #ef4444, #f87171);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.sentiment-neutral {
    background: linear-gradient(135deg, #f59e0b, #fbbf24);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.confidence-row {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-top: 14px;
}
.confidence-bar-track {
    flex: 1;
    height: 6px;
    background: #1e293b;
    border-radius: 3px;
    overflow: hidden;
}
.confidence-bar-fill {
    height: 6px;
    border-radius: 3px;
}
.fill-positive { background: linear-gradient(90deg, #10b981, #34d399); }
.fill-negative { background: linear-gradient(90deg, #ef4444, #f87171); }
.fill-neutral  { background: linear-gradient(90deg, #f59e0b, #fbbf24); }
.confidence-pct {
    font-size: 13px;
    font-family: 'DM Mono', monospace;
    color: #94a3b8;
    white-space: nowrap;
    font-weight: 500;
}
.model-type-badge {
    display: inline-block;
    font-size: 11px;
    font-family: 'DM Mono', monospace;
    background: rgba(59,130,246,0.08);
    border: 1px solid rgba(59,130,246,0.2);
    border-radius: 8px;
    padding: 4px 10px;
    color: #60a5fa;
    margin-top: 12px;
}

/* ── SECTION HEADING ── */
.section-heading {
    font-size: 15px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #e2e8f0;
    margin-bottom: 1.2rem;
    border-left: 3px solid #3b82f6;
    padding-left: 12px;
}

/* ── AGREEMENT BANNER ── */
.agreement-banner {
    border-radius: 12px;
    padding: 14px 18px;
    font-size: 14px;
    font-weight: 500;
    margin-top: 1.2rem;
    display: flex;
    align-items: center;
    gap: 12px;
}
.banner-agree {
    background: rgba(16,185,129,0.08);
    border: 1px solid rgba(16,185,129,0.25);
    color: #34d399;
}
.banner-disagree {
    background: rgba(245,158,11,0.08);
    border: 1px solid rgba(245,158,11,0.25);
    color: #fbbf24;
}
.banner-warn {
    background: rgba(239,68,68,0.08);
    border: 1px solid rgba(239,68,68,0.25);
    color: #f87171;
}

/* ── TOKEN DISPLAY ── */
.token-grid {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin-top: 12px;
}
.token {
    font-family: 'DM Mono', monospace;
    font-size: 12px;
    background: rgba(59,130,246,0.08);
    border: 1px solid rgba(59,130,246,0.2);
    border-radius: 8px;
    padding: 4px 12px;
    color: #7dd3fc;
}
.word-stat {
    font-size: 12px;
    color: #475569;
    margin-top: 12px;
    font-family: 'DM Mono', monospace;
}

/* ── FOOTER ── */
.sentiscope-footer {
    text-align: center;
    font-size: 12px;
    color: #334155;
    margin-top: 3rem;
    font-family: 'DM Mono', monospace;
    letter-spacing: 0.05em;
    border-top: 1px solid #0f172a;
    padding-top: 1.5rem;
    line-height: 2;
}
.sentiscope-footer span { color: #3b82f6; }
</style>
""", unsafe_allow_html=True)

# ── LOAD MODEL ─────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model = pickle.load(open('model.pkl', 'rb'))
    tfidf = pickle.load(open('tfidf.pkl', 'rb'))
    return model, tfidf

model, tfidf = load_model()
vader        = SentimentIntensityAnalyzer()

# ── CLEAN TEXT ─────────────────────────────────────────────────────────
stop_words = set(stopwords.words('english'))
stemmer    = PorterStemmer()

def clean_text(text):
    text  = text.lower()
    text  = re.sub(r'[^a-z\s]', ' ', text)
    words = text.split()
    words = [stemmer.stem(w) for w in words
             if w not in stop_words and len(w) > 2]
    return ' '.join(words)

# ── HEADER ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="sentiscope-header">
    <div class="sentiscope-logo">🔬</div>
    <h1 class="sentiscope-title">SentiScope</h1>
    <span class="version-chip">v1.0 · NLP</span>
</div>
<p class="sentiscope-subtitle">
    Review Intelligence Platform &nbsp;·&nbsp;
    Trained on 50,000 Amazon Reviews &nbsp;·&nbsp;
    ML + Rule-based Dual Analysis
</p>
""", unsafe_allow_html=True)

# ── STATS ROW ──────────────────────────────────────────────────────────
st.markdown("""
<div class="stats-row">
    <div class="stat-card">
        <div class="stat-value">87.5%</div>
        <div class="stat-label">Model Accuracy</div>
    </div>
    <div class="stat-card">
        <div class="stat-value">50K</div>
        <div class="stat-label">Reviews Trained</div>
    </div>
    <div class="stat-card">
        <div class="stat-value">2</div>
        <div class="stat-label">Analysis Models</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── INPUT ──────────────────────────────────────────────────────────────
st.markdown('<div class="input-label">📝 &nbsp; Input Review</div>', unsafe_allow_html=True)

user_input = st.text_area(
    label="",
    height=150,
    placeholder="Paste any Amazon product review here — e.g. 'The battery life on this is incredible. Lasts 3 days easily. Best phone I have ever owned.'",
    label_visibility="collapsed"
)

col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
with col_btn2:
    analyse_btn = st.button("🔍 &nbsp; Analyse Sentiment", use_container_width=True)

# ── RESULTS ────────────────────────────────────────────────────────────
if analyse_btn:
    if not user_input.strip():
        st.markdown("""
        <div class="agreement-banner banner-warn">
            ⚠️ &nbsp; Please enter a review before analysing.
        </div>
        """, unsafe_allow_html=True)
    else:
        # ML Prediction
        cleaned      = clean_text(user_input)
        vec          = tfidf.transform([cleaned])
        ml_pred      = model.predict(vec)[0]
        ml_proba     = model.predict_proba(vec)[0]
        ml_label     = "Positive" if ml_pred == 1 else "Negative"
        ml_conf      = max(ml_proba) * 100
        ml_class     = "positive" if ml_pred == 1 else "negative"
        ml_sent_class= "sentiment-positive" if ml_pred == 1 else "sentiment-negative"
        ml_fill      = "fill-positive" if ml_pred == 1 else "fill-negative"
        ml_icon      = "↑" if ml_pred == 1 else "↓"

        # VADER Prediction
        v_scores   = vader.polarity_scores(user_input)
        v_compound = v_scores['compound']
        if v_compound >= 0.05:
            v_label = "Positive"; v_class = "positive"
            v_sent_class = "sentiment-positive"; v_fill = "fill-positive"
            v_icon = "↑"
        elif v_compound <= -0.05:
            v_label = "Negative"; v_class = "negative"
            v_sent_class = "sentiment-negative"; v_fill = "fill-negative"
            v_icon = "↓"
        else:
            v_label = "Neutral"; v_class = "neutral"
            v_sent_class = "sentiment-neutral"; v_fill = "fill-neutral"
            v_icon = "→"

        st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)

        # ── RESULT CARDS ──────────────────────────────────────────────
        col1, col2 = st.columns(2, gap="medium")

        with col1:
            st.markdown(f"""
            <div class="result-card {ml_class}">
                <div class="result-card-label">🤖 &nbsp; ML Model &nbsp;·&nbsp; Logistic Regression</div>
                <div class="result-sentiment {ml_sent_class}">{ml_icon} {ml_label}</div>
                <div class="confidence-row">
                    <div class="confidence-bar-track">
                        <div class="confidence-bar-fill {ml_fill}" style="width:{ml_conf:.1f}%"></div>
                    </div>
                    <span class="confidence-pct">{ml_conf:.1f}%</span>
                </div>
                <div class="model-type-badge">TF-IDF · Logistic Regression · scikit-learn</div>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div class="result-card {v_class}">
                <div class="result-card-label">📏 &nbsp; VADER &nbsp;·&nbsp; Rule-based NLP</div>
                <div class="result-sentiment {v_sent_class}">{v_icon} {v_label}</div>
                <div class="confidence-row">
                    <div class="confidence-bar-track">
                        <div class="confidence-bar-fill {v_fill}" style="width:{min(abs(v_compound)*100,100):.1f}%"></div>
                    </div>
                    <span class="confidence-pct">score: {v_compound:+.3f}</span>
                </div>
                <div class="model-type-badge">Rule-based · Lexicon · vaderSentiment</div>
            </div>
            """, unsafe_allow_html=True)

        # ── AGREEMENT BANNER ──────────────────────────────────────────
        if ml_label == v_label:
            st.markdown(f"""
            <div class="agreement-banner banner-agree">
                ✅ &nbsp; Both models agree — this review is <strong>{ml_label}</strong>. High confidence result.
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="agreement-banner banner-disagree">
                ⚡ &nbsp; Models disagree — ML predicts <strong>{ml_label}</strong>,
                VADER predicts <strong>{v_label}</strong>.
                ML model is more reliable for product reviews.
            </div>
            """, unsafe_allow_html=True)

        # ── COMPARISON CHART ──────────────────────────────────────────
        st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)
        st.markdown('<div class="section-heading">📊 &nbsp; Model Comparison</div>', unsafe_allow_html=True)

        fig = go.Figure()
        fig.add_trace(go.Bar(
            name='ML Model',
            x=['Positive', 'Negative', 'Neutral'],
            y=[ml_proba[1]*100, ml_proba[0]*100, 0],
            marker=dict(color=['#10b981','#ef4444','#f59e0b'], opacity=1.0, line=dict(width=0)),
            width=0.25
        ))
        fig.add_trace(go.Bar(
            name='VADER',
            x=['Positive', 'Negative', 'Neutral'],
            y=[v_scores['pos']*100, v_scores['neg']*100, v_scores['neu']*100],
            marker=dict(color=['#10b981','#ef4444','#f59e0b'], opacity=0.35, line=dict(width=0)),
            width=0.25
        ))
        fig.update_layout(
            barmode='group',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(family='DM Sans', color='#64748b', size=13),
            yaxis=dict(
                title='Score (%)',
                gridcolor='#0f172a',
                zerolinecolor='#1e293b',
                tickfont=dict(family='DM Mono', size=12, color='#475569'),
                range=[0, 110]
            ),
            xaxis=dict(
                gridcolor='rgba(0,0,0,0)',
                tickfont=dict(family='DM Sans', size=13, color='#94a3b8')
            ),
            legend=dict(
                bgcolor='rgba(15,23,42,0.8)',
                bordercolor='#1e293b',
                borderwidth=1,
                font=dict(size=12, color='#94a3b8')
            ),
            height=320,
            margin=dict(l=10, r=10, t=10, b=10),
            bargap=0.4,
            bargroupgap=0.08
        )
        st.plotly_chart(fig, use_container_width=True)

        # ── TOKEN DISPLAY ─────────────────────────────────────────────
        st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)
        st.markdown('<div class="section-heading">🔤 &nbsp; Tokenised Input</div>', unsafe_allow_html=True)

        if cleaned.strip():
            words          = cleaned.split()
            token_html     = ''.join([f'<span class="token">{w}</span>' for w in words])
            original_count = len(user_input.split())
            cleaned_count  = len(words)
            removed        = original_count - cleaned_count
            reduction_pct  = (removed / original_count * 100) if original_count > 0 else 0
            st.markdown(f"""
            <div class="token-grid">{token_html}</div>
            <div class="word-stat">
                {original_count} words input &nbsp;→&nbsp;
                {cleaned_count} tokens analysed &nbsp;·&nbsp;
                {removed} noise words removed ({reduction_pct:.0f}% reduction)
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="agreement-banner banner-warn">
                No meaningful tokens found after cleaning.
            </div>
            """, unsafe_allow_html=True)

# ── FOOTER ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="sentiscope-footer">
    <span>SentiScope</span> · Built with Python · scikit-learn · NLTK · VADER · Streamlit · Plotly
    <br>
    Trained on 50,000 Amazon Reviews · TF-IDF + Logistic Regression · 87.5% Accuracy
</div>
""", unsafe_allow_html=True)