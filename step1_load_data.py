import bz2
import pandas as pd

def load_data(filepath, num_rows=50000):
    labels  = []
    reviews = []

    with bz2.open(filepath, 'rt', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= num_rows:
                break
            label, _, text = line.partition(' ')
            labels.append(1 if label == '__label__2' else 0)
            reviews.append(text.strip())

    return pd.DataFrame({'review': reviews, 'sentiment': labels})


df = load_data('train.ft.txt.bz2', num_rows=50000)

print("First 3 reviews:")
print(df.head(3))
print(f"\nDataset size: {df.shape[0]} rows")
print(f"Positive reviews: {df['sentiment'].sum()}")
print(f"Negative reviews: {df.shape[0] - df['sentiment'].sum()}")


df.to_csv('raw_reviews.csv', index=False)
print("✓ Saved as raw_reviews.csv - open this in Excel")