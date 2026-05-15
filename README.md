# 🔬 SentiScope — Amazon Review Sentiment Analyser

> Dual-model sentiment analysis platform trained on 50,000 Amazon reviews.
> Compares ML-based prediction against VADER rule-based NLP in real time.

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange?style=flat-square&logo=scikit-learn)
![Streamlit](https://img.shields.io/badge/Streamlit-deployed-red?style=flat-square&logo=streamlit)
![Accuracy](https://img.shields.io/badge/Accuracy-87.5%25-green?style=flat-square)

---

## 🔗 Live Demo
**[→ Try SentiScope here](https://sentiscope.streamlit.app)**

---

## 📌 What This Project Does

SentiScope takes any Amazon product review as input and:
- Predicts sentiment as **Positive** or **Negative** with confidence score
- Runs two independent models — ML-based and rule-based — simultaneously
- Compares both results side by side with a visual chart
- Shows exactly which tokens the model analysed after cleaning

---

## 🧠 How It Works

```
Raw Review Text
      ↓
Text Cleaning (lowercase → remove punctuation → remove stopwords → stemming)
      ↓
TF-IDF Vectorisation (text → 10,000 numbers)
      ↓
Logistic Regression Model (trained on 40,000 reviews)
      ↓
Sentiment Prediction + Confidence Score
```

Simultaneously — VADER analyses the raw text using its 7,500-word lexicon
and outputs a compound score from -1 to +1.

Both results are displayed side by side for comparison.

---

## 📊 Model Performance

| Metric | Score |
|--------|-------|
| Accuracy | 87.5% |
| Positive Precision | 87% |
| Negative Precision | 88% |
| Positive Recall | 89% |
| Negative Recall | 86% |
| Training samples | 40,000 |
| Test samples | 10,000 |

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| Python 3.10+ | Core language |
| pandas | Data loading and manipulation |
| NLTK | Stopwords, stemming, text preprocessing |
| scikit-learn | TF-IDF vectorisation + Logistic Regression |
| vaderSentiment | Rule-based sentiment baseline |
| Streamlit | Web application framework |
| Plotly | Interactive comparison charts |
| pickle | Model serialisation |

---

## 📁 Project Structure

```
sentiscope/
│
├── app.py                  ← Main Streamlit application
├── step1_load_data.py      ← Data loading pipeline
├── step2_clean.py          ← Text preprocessing pipeline
├── step3_train_model.py    ← Model training + evaluation
├── eda.py                  ← Exploratory data analysis
├── inspect_model.py        ← Model inspection utility
│
├── model.pkl               ← Trained Logistic Regression model
├── tfidf.pkl               ← Fitted TF-IDF vectoriser
│
└── requirements.txt        ← Python dependencies
```

---

## 🚀 Run Locally

```bash
# Clone the repository
git clone https://github.com/rudravamsi/sentiscope.git
cd sentiscope

# Install dependencies
pip install -r requirements.txt

# Run the app
python -m streamlit run app.py
```

---

## 💡 Key Learnings

- End-to-end NLP pipeline from raw text to deployed web app
- Difference between statistical ML and rule-based NLP approaches
- Why balanced datasets matter for unbiased model training
- How TF-IDF converts text into numerical features
- Real limitations of bag-of-words models — sarcasm, negation, context

---

## 🔮 Future Improvements

- Fine-tune a BERT transformer for context-aware predictions
- Add multi-class sentiment (1–5 star rating prediction)
- Handle sarcasm and negation with advanced NLP techniques
- Add batch analysis — upload a CSV of reviews and analyse all at once

---

## 👤 Author

**Rudra** — BTech CSE Final Year  
Built as part of an AI/ML portfolio project

---

*Trained on the Amazon Reviews dataset · 50,000 reviews · 87.5% test accuracy*
