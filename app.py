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
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap');

/* ── GLOBAL RESET ── */
*, *::before, *::after { box-sizing: border-box; }

html, body, .stApp {
    background-color: #0b0f19 !important;
    font-family: 'DM Sans', sans-serif !important;
    color: #e2e8f0 !important;
}

/* ── HIDE STREAMLIT DEFAULTS ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container {
    padding: 2rem 3rem !important;
    max-width: 1200px !important;
}

/* ── CUSTOM HEADER ── */
.sentiscope-header {
    display: flex;
    align-items: center;
    gap: 14px;
    margin-bottom: 6px;
}
.sentiscope-logo {
    width: 44px;
    height: 44px;
    background: linear-gradient(135deg, #3b82f6, #8b5cf6);
    border-radius: 12px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 22px;
}
.sentiscope-title {
    font-size: 28px;
    font-weight: 600;
    background: linear-gradient(135deg, #3b82f6, #a78bfa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0;
    letter-spacing: -0.5px;
}
.sentiscope-subtitle {
    font-size: 14px;
    color: #64748b;
    margin: 0 0 2rem 0;
    font-weight: 400;
    letter-spacing: 0.02em;
}
.version-chip {
    display: inline-block;
    font-size: 11px;
    font-family: 'DM Mono', monospace;
    background: #1e293b;
    color: #64748b;
    border: 1px solid #1e3a5f;
    border-radius: 20px;
    padding: 2px 10px;
    margin-left: 10px;
    vertical-align: middle;
}

/* ── DIVIDER ── */
.custom-divider {
    height: 1px;
    background: linear-gradient(90deg, #1e3a5f, #312e81, #1e3a5f);
    margin: 1.5rem 0;
    border: none;
}

/* ── INPUT CARD ── */
.input-card {
    background: #111827;
    border: 1px solid #1e293b;
    border-radius: 16px;
    padding: 1.5rem;
    margin-bottom: 1.5rem;
}
.input-label {
    font-size: 12px;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #3b82f6;
    margin-bottom: 10px;
}

/* ── TEXTAREA STYLING ── */
.stTextArea textarea {
    background: #0b0f19 !important;
    border: 1px solid #1e293b !important;
    border-radius: 10px !important;
    color: #e2e8f0 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 14px !important;
    padding: 14px !important;
    resize: vertical !important;
    transition: border-color 0.2s !important;
}
.stTextArea textarea:focus {
    border-color: #3b82f6 !important;
    box-shadow: 0 0 0 3px rgba(59,130,246,0.1) !important;
}
.stTextArea textarea::placeholder { color: #334155 !important; }

/* ── BUTTON ── */
.stButton > button {
    background: linear-gradient(135deg, #2563eb, #7c3aed) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 14px !important;
    font-weight: 500 !important;
    padding: 0.65rem 1.5rem !important;
    width: 100% !important;
    transition: all 0.2s !important;
    letter-spacing: 0.02em !important;
}
.stButton > button:hover {
    opacity: 0.9 !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 8px 25px rgba(37,99,235,0.3) !important;
}

/* ── RESULT CARDS ── */
.result-card {
    background: #111827;
    border: 1px solid #1e293b;
    border-radius: 16px;
    padding: 1.4rem 1.6rem;
    height: 100%;
    position: relative;
    overflow: hidden;
}
.result-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
}
.result-card.positive::before {
    background: linear-gradient(90deg, #10b981, #34d399);
}
.result-card.negative::before {
    background: linear-gradient(90deg, #ef4444, #f87171);
}
.result-card.neutral::before {
    background: linear-gradient(90deg, #f59e0b, #fbbf24);
}
.result-card-label {
    font-size: 11px;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #475569;
    margin-bottom: 12px;
}
.result-sentiment {
    font-size: 32px;
    font-weight: 600;
    margin: 4px 0;
    letter-spacing: -0.5px;
}
.sentiment-positive { color: #10b981; }
.sentiment-negative { color: #ef4444; }
.sentiment-neutral  { color: #f59e0b; }
.confidence-row {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-top: 12px;
}
.confidence-bar-track {
    flex: 1;
    height: 4px;
    background: #1e293b;
    border-radius: 2px;
    overflow: hidden;
}
.confidence-bar-fill {
    height: 4px;
    border-radius: 2px;
    transition: width 0.8s ease;
}
.fill-positive { background: linear-gradient(90deg, #10b981, #34d399); }
.fill-negative { background: linear-gradient(90deg, #ef4444, #f87171); }
.fill-neutral  { background: linear-gradient(90deg, #f59e0b, #fbbf24); }
.confidence-pct {
    font-size: 12px;
    font-family: 'DM Mono', monospace;
    color: #64748b;
    white-space: nowrap;
}
.model-type-badge {
    display: inline-block;
    font-size: 10px;
    font-family: 'DM Mono', monospace;
    background: #0f172a;
    border: 1px solid #1e293b;
    border-radius: 6px;
    padding: 2px 8px;
    color: #475569;
    margin-top: 8px;
}

/* ── CHART SECTION ── */
.chart-card {
    background: #111827;
    border: 1px solid #1e293b;
    border-radius: 16px;
    padding: 1.4rem 1.6rem;
    margin-top: 1.2rem;
}
.section-heading {
   
    font-size: 12px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #94a3b8;
    margin-bottom: 1rem;
    border-left: 3px solid #3b82f6;
    padding-left: 10px;
}

/* ── TOKEN DISPLAY ── */
.token-grid {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
    margin-top: 10px;
}
.token {
    font-family: 'DM Mono', monospace;
    font-size: 12px;
    background: #0f172a;
    border: 1px solid #1e293b;
    border-radius: 6px;
    padding: 3px 10px;
    color: #7dd3fc;
}
.word-stat {
    font-size: 12px;
    color: #475569;
    margin-top: 10px;
    font-family: 'DM Mono', monospace;
}

/* ── AGREEMENT BANNER ── */
.agreement-banner {
    border-radius: 10px;
    padding: 12px 16px;
    font-size: 13px;
    font-weight: 500;
    margin-top: 1rem;
    display: flex;
    align-items: center;
    gap: 10px;
}
.banner-agree {
    background: rgba(16,185,129,0.08);
    border: 1px solid rgba(16,185,129,0.2);
    color: #10b981;
}
.banner-disagree {
    background: rgba(245,158,11,0.08);
    border: 1px solid rgba(245,158,11,0.2);
    color: #f59e0b;
}

/* ── STATS ROW ── */
.stats-row {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 12px;
    margin-bottom: 1.5rem;
}
.stat-card {
    background: #111827;
    border: 1px solid #1e293b;
    border-radius: 12px;
    padding: 14px 16px;
    text-align: center;
}
.stat-value {
    font-size: 22px;
    font-weight: 600;
    color: #e2e8f0;
    font-family: 'DM Mono', monospace;
    letter-spacing: -0.5px;
}
.stat-label {
    font-size: 11px;
    color: #475569;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-top: 3px;
}

/* ── FOOTER ── */
.sentiscope-footer {
   
    text-align: center;
    font-size: 11px;
    color: #475569;
    margin-top: 3rem;
    font-family: 'DM Mono', monospace;
    letter-spacing: 0.05em;
    border-top: 1px solid #1e293b;
    padding-top: 1.5rem;
    line-height: 1.8;
}

/* ── STREAMLIT METRIC OVERRIDE ── */
[data-testid="stMetric"] {
    background: transparent !important;
}
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
    <span class="version-chip">v1.0</span>
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
    height=140,
    placeholder="Paste any product review here — e.g. 'The battery life on this is incredible. Lasts 3 days easily. Best phone I have ever owned.'",
    label_visibility="collapsed"
)

col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
with col_btn2:
    analyse_btn = st.button("🔍 &nbsp; Analyse Sentiment", use_container_width=True)

# ── RESULTS ────────────────────────────────────────────────────────────
if analyse_btn:
    if not user_input.strip():
        st.markdown("""
        <div class="agreement-banner banner-disagree">
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
            v_conf = v_scores['pos'] * 100; v_icon = "↑"
        elif v_compound <= -0.05:
            v_label = "Negative"; v_class = "negative"
            v_sent_class = "sentiment-negative"; v_fill = "fill-negative"
            v_conf = v_scores['neg'] * 100; v_icon = "↓"
        else:
            v_label = "Neutral"; v_class = "neutral"
            v_sent_class = "sentiment-neutral"; v_fill = "fill-neutral"
            v_conf = 50.0; v_icon = "→"

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
                        <div class="confidence-bar-fill {v_fill}" style="width:{min(abs(v_compound)*100, 100):.1f}%"></div>
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
                ⚡ &nbsp; Models disagree — ML predicts <strong>{ml_label}</strong>, VADER predicts <strong>{v_label}</strong>. 
                ML model is more reliable for product reviews.
            </div>
            """, unsafe_allow_html=True)

        # ── COMPARISON CHART ──────────────────────────────────────────
        st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)
        st.markdown('<div class="section-heading">Model Comparison</div>', unsafe_allow_html=True)

        fig = go.Figure()

        fig.add_trace(go.Bar(
            name='ML Model',
            x=['Positive', 'Negative', 'Neutral'],
            y=[
                ml_proba[1] * 100,
                ml_proba[0] * 100,
                0
            ],
            marker=dict(
                color=['#10b981', '#ef4444', '#f59e0b'],
                opacity=0.9,
                line=dict(width=0)
            ),
            width=0.3
        ))

        fig.add_trace(go.Bar(
            name='VADER',
            x=['Positive', 'Negative', 'Neutral'],
            y=[
                v_scores['pos'] * 100,
                v_scores['neg'] * 100,
                v_scores['neu'] * 100
            ],
            marker=dict(
                color=['#10b981', '#ef4444', '#f59e0b'],
                opacity=0.4,
                line=dict(width=0)
            ),
            width=0.3
        ))

        fig.update_layout(
            barmode='group',
            paper_bgcolor='#111827',
            plot_bgcolor='#111827',
            font=dict(family='DM Sans', color='#64748b', size=12),
            yaxis=dict(
                title='Score (%)',
                gridcolor='#1e293b',
                zerolinecolor='#1e293b',
                tickfont=dict(family='DM Mono', size=11),
                range=[0, 105]
            ),
            xaxis=dict(
                gridcolor='#1e293b',
                tickfont=dict(family='DM Sans', size=12, color='#94a3b8')
            ),
            legend=dict(
                bgcolor='#0f172a',
                bordercolor='#1e293b',
                borderwidth=1,
                font=dict(size=11)
            ),
            height=300,
            margin=dict(l=10, r=10, t=20, b=10),
            bargap=0.4,
            bargroupgap=0.1
        )

        st.plotly_chart(fig, use_container_width=True)

        # ── WORD TOKEN DISPLAY ────────────────────────────────────────
        st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)
        st.markdown('<div class="section-heading">Tokenised Input</div>', unsafe_allow_html=True)

        if cleaned.strip():
            words      = cleaned.split()
            token_html = ''.join([f'<span class="token">{w}</span>' for w in words])
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
            <div class="agreement-banner banner-disagree">
                No meaningful tokens found after cleaning.
            </div>
            """, unsafe_allow_html=True)

# ── FOOTER ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="sentiscope-footer">
    SentiScope · Built with Python · scikit-learn · NLTK · VADER · Streamlit · Plotly
    <br>Trained on 50,000 Amazon Reviews · TF-IDF + Logistic Regression · 87.5% Accuracy
</div>
""", unsafe_allow_html=True)