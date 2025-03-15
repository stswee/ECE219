import pandas as pd
# Load the preprocessed tweets CSV file
input_csv = "preprocessed_tweets.csv"

df = pd.read_csv(input_csv, engine='python')

df = df[df["tweet_text"].notna() & (df["tweet_text"] != "")]
df = df.dropna()

subsample = df[
    (df['posting_time'].str.startswith('2015-02-01')) &
    (df['tweet_text'].str.len() <= 150)
]

subsample.to_csv("subsample_preprocessed_tweets.csv")