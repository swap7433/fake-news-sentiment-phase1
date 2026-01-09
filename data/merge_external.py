# merge_external.py
import pandas as pd
a = pd.read_csv("data/live_tweets.csv")
b = pd.read_csv("data/external_fake.csv")   # downloaded from external source
b = b.rename(columns={'text':'content'})    # adapt if needed
a2 = pd.concat([a, b[['content','label']]], ignore_index=True)
a2.to_csv("data/live_tweets.csv", index=False)
print("Merged, new counts:")
print(a2['label'].value_counts())
