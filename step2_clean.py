import bz2
import re
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download('stopwords', quiet=True)

def load_data(filepath, num_rows=50000):
    labels, reviews = [], []
    with bz2.open(filepath, 'rt', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= num_rows:
                break
            label, _, text = line.partition(' ')
            labels.append(1 if label == '__label__2' else 0)
            reviews.append(text.strip())
    return pd.DataFrame({'review': reviews, 'sentiment': labels})

df = load_data('train.ft.txt.bz2')

stop_words = set(stopwords.words('english'))
stemmer    = PorterStemmer()

def clean_text(text):
    text  = text.lower()
    text  = re.sub(r'[^a-z\s]', ' ', text)
    words = text.split()
    words = [stemmer.stem(w) for w in words
             if w not in stop_words and len(w) > 2]
    return ' '.join(words)

print("Cleaning 50,000 reviews... wait ~30 seconds")
df['clean_review'] = df['review'].apply(clean_text)

print("\nBEFORE cleaning:")
print(df['review'][0])
print("\nAFTER cleaning:")
print(df['clean_review'][0])

df.to_csv('cleaned_reviews.csv', index=False)
print("\n✓ Saved as cleaned_reviews.csv")
print(f"✓ Total rows saved: {len(df)}")