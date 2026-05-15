import pickle
import numpy as np

# ── Open model.pkl ─────────────────────────────────────────
model = pickle.load(open('model.pkl', 'rb'))

print("=" * 50)
print("MODEL.PKL — What's inside")
print("=" * 50)
print(f"Model type        : {type(model)}")
print(f"Max iterations    : {model.max_iter}")
print(f"Classes learned   : {model.classes_}")
print(f"Total features    : {model.coef_.shape[1]}")

# ── Open tfidf.pkl ─────────────────────────────────────────
tfidf = pickle.load(open('tfidf.pkl', 'rb'))

print("\n" + "=" * 50)
print("TFIDF.PKL — What's inside")
print("=" * 50)
print(f"Vectoriser type   : {type(tfidf)}")
print(f"Total vocabulary  : {len(tfidf.vocabulary_)}")
print(f"Max features set  : {tfidf.max_features}")

# ── Show sample vocabulary words ───────────────────────────
print("\n" + "=" * 50)
print("SAMPLE WORDS IN VOCABULARY (first 20)")
print("=" * 50)
words = list(tfidf.vocabulary_.keys())
print(words[:20])

# ── Show most POSITIVE words the model learned ─────────────
print("\n" + "=" * 50)
print("TOP 15 WORDS THAT SCREAM POSITIVE")
print("=" * 50)
feature_names = tfidf.get_feature_names_out()
coefficients  = model.coef_[0]
top_positive  = np.argsort(coefficients)[-15:][::-1]
for i in top_positive:
    print(f"  {feature_names[i]:<20} score: {coefficients[i]:.4f}")

# ── Show most NEGATIVE words the model learned ─────────────
print("\n" + "=" * 50)
print("TOP 15 WORDS THAT SCREAM NEGATIVE")
print("=" * 50)
top_negative = np.argsort(coefficients)[:15]
for i in top_negative:
    print(f"  {feature_names[i]:<20} score: {coefficients[i]:.4f}")