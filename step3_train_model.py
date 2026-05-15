import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# ── LOAD cleaned data ────────────────────────────────────────
print("Loading cleaned data...")
df = pd.read_csv('cleaned_reviews.csv')
df = df.dropna()  # remove any empty rows
print(f"✓ Loaded {len(df)} reviews")

# ── STEP A: Split into train and test ────────────────────────
print("\nSplitting into train and test sets...")
X_train, X_test, y_train, y_test = train_test_split(
    df['clean_review'],  # input  → the cleaned review text
    df['sentiment'],     # output → 1 or 0
    test_size=0.2,       # 20% for testing
    random_state=42      # fixed number so results are same every run
)
print(f"✓ Training on : {len(X_train)} reviews")
print(f"✓ Testing on  : {len(X_test)} reviews")

# ── STEP B: Convert text to numbers using TF-IDF ─────────────
print("\nConverting text to numbers using TF-IDF...")
tfidf = TfidfVectorizer(max_features=10000)

# fit_transform on train → learn vocabulary + convert to numbers
X_train_tfidf = tfidf.fit_transform(X_train)

# transform only on test → use same vocabulary, just convert
X_test_tfidf = tfidf.transform(X_test)

print(f"✓ Each review is now {X_train_tfidf.shape[1]} numbers")
print(f"✓ Training matrix size: {X_train_tfidf.shape}")

# ── STEP C: Train the model ───────────────────────────────────
print("\nTraining Logistic Regression model...")
model = LogisticRegression(max_iter=200)
model.fit(X_train_tfidf, y_train)
print("✓ Training complete!")

# ── STEP D: Evaluate the model ────────────────────────────────
print("\nEvaluating on test set...")
y_pred = model.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, y_pred)
print(f"\n✓ Accuracy: {accuracy * 100:.2f}%")
print("\nDetailed Report:")
print(classification_report(y_test, y_pred,
      target_names=['Negative', 'Positive']))

# ── STEP E: Save model and vectoriser ─────────────────────────
pickle.dump(model, open('model.pkl', 'wb'))
pickle.dump(tfidf, open('tfidf.pkl', 'wb'))
print("✓ model.pkl saved")
print("✓ tfidf.pkl saved")

# ── STEP F: Test with your own sentences ─────────────────────
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    words = text.split()
    words = [stemmer.stem(w) for w in words
             if w not in stop_words and len(w) > 2]
    return ' '.join(words)

def predict(text):
    cleaned = clean_text(text)
    vec = tfidf.transform([cleaned])
    result = model.predict(vec)[0]
    prob = model.predict_proba(vec)[0]
    label = "POSITIVE" if result == 1 else "NEGATIVE"
    confidence = max(prob) * 100
    print(f"  → {label}  ({confidence:.1f}% confident)")

print("\n" + "="*50)
print("TESTING WITH CUSTOM REVIEWS")
print("="*50)
predict("This product is absolutely amazing, works perfectly!")
predict("Terrible quality, broke after 2 days. Complete waste of money.")
predict("It is okay, nothing special, average product.")
predict("Best purchase I have ever made, highly recommend!")
predict("Stopped working after one week. Very disappointed.")