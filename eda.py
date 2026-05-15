import bz2
import pandas as pd

# ── Load data ──────────────────────────────────────────────
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

# ── 1. SHAPE ────────────────────────────────────────────────
print("=" * 50)
print("SHAPE")
print("=" * 50)
print(f"Rows    : {df.shape[0]}")   # how many reviews
print(f"Columns : {df.shape[1]}")   # how many columns
print(f"Columns names : {list(df.columns)}")

# ── 2. FIRST AND LAST ROWS ──────────────────────────────────
print("\n" + "=" * 50)
print("FIRST 5 ROWS")
print("=" * 50)
print(df.head(5))

print("\n" + "=" * 50)
print("LAST 5 ROWS")
print("=" * 50)
print(df.tail(5))

# ── 3. DATA TYPES ───────────────────────────────────────────
print("\n" + "=" * 50)
print("COLUMN DATA TYPES")
print("=" * 50)
print(df.dtypes)

# ── 4. MISSING VALUES ───────────────────────────────────────
print("\n" + "=" * 50)
print("MISSING VALUES (empty cells)")
print("=" * 50)
print(df.isnull().sum())

# ── 5. LABEL DISTRIBUTION ───────────────────────────────────
print("\n" + "=" * 50)
print("LABEL DISTRIBUTION")
print("=" * 50)
pos = df['sentiment'].sum()
neg = df.shape[0] - pos
print(f"Positive reviews (1) : {pos}  ({pos/len(df)*100:.1f}%)")
print(f"Negative reviews (0) : {neg}  ({neg/len(df)*100:.1f}%)")

# ── 6. REVIEW LENGTH ANALYSIS ───────────────────────────────
print("\n" + "=" * 50)
print("REVIEW LENGTH (in words)")
print("=" * 50)
df['word_count'] = df['review'].apply(lambda x: len(x.split()))
print(f"Shortest review : {df['word_count'].min()} words")
print(f"Longest review  : {df['word_count'].max()} words")
print(f"Average length  : {df['word_count'].mean():.0f} words")
print(f"Most common     : {df['word_count'].mode()[0]} words")

# ── 7. SAMPLE REVIEWS ───────────────────────────────────────
print("\n" + "=" * 50)
print("SAMPLE POSITIVE REVIEW")
print("=" * 50)
print(df[df['sentiment']==1]['review'].iloc[0])

print("\n" + "=" * 50)
print("SAMPLE NEGATIVE REVIEW")
print("=" * 50)
print(df[df['sentiment']==0]['review'].iloc[0])