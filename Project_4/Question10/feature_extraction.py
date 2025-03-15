import pandas as pd
import numpy as np
import os
import re
import nltk
from nltk.tokenize import word_tokenize
nltk.download("vader_lexicon")
from nltk.stem import WordNetLemmatizer
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
import os
import torch
torch.set_num_threads(32)

# Define file paths
input_csv = "subsample_preprocessed_tweets.csv"
output_dir = "extracted_features"
os.makedirs(output_dir, exist_ok=True)

# Load the preprocessed tweets CSV file
print("Loading Data")
df = pd.read_csv(input_csv)

# --- Extract TF-IDF Features ---

def lemmatize_text(text: str) -> str:
    lemmatizer = WordNetLemmatizer()
    # Tokenize text
    tokens = word_tokenize(text)
    # Lemmatize each token and lower-case them
    lemmatized_tokens = [lemmatizer.lemmatize(token.lower()) for token in tokens]
    return " ".join(lemmatized_tokens)

print("Lemmatizing tweet texts...")
df["tweet_text_lemmatized"] = df["tweet_text"].apply(lemmatize_text)

print("Extracting TF-IDF features...")
tfidf_vectorizer = TfidfVectorizer(max_features=1000)
tfidf_features = tfidf_vectorizer.fit_transform(df["tweet_text_lemmatized"]).toarray()
print("TF-IDF features shape:", tfidf_features.shape)

# Save TF-IDF features and vocabulary
np.save(os.path.join(output_dir, "tfidf_features.npy"), tfidf_features)
vocab_df = pd.DataFrame(list(tfidf_vectorizer.vocabulary_.items()), columns=["term", "index"])
vocab_df.to_csv(os.path.join(output_dir, "tfidf_vocabulary.csv"), index=False)


# --- Extract Transformer (SBERT) Features ---
print("Extracting SBERT features...")
sbert_model = SentenceTransformer('all-MiniLM-L6-v2', device='cuda')
sbert_features = sbert_model.encode(df["tweet_text"].tolist(), batch_size=32, show_progress_bar=True)
print("SBERT features shape:", sbert_features.shape)

# Save SBERT features
np.save(os.path.join(output_dir, "sbert_features.npy"), sbert_features)

# --- Save Metadata ---
# Initialize VADER sentiment analyzer
sia = SentimentIntensityAnalyzer()

# Compute additional metadata features
df["vader_sentiment"] = df["tweet_text"].apply(lambda x: sia.polarity_scores(x)["compound"] if pd.notnull(x) else 0)
df["tweet_text_length"] = df["tweet_text"].str.len()
df["num_hashtags"] = df["hashtags"].apply(lambda x: len(x.split(',')) if pd.notnull(x) and x.strip() != "" else 0)

# Define metadata columns (Fixed missing comma issue)
metadata_cols = [
    "unix_time", "retweet_count", "favorite_count", "num_hashtags",
    "user_followers_count", "user_friends_count", "user_statuses_count",
    "vader_sentiment", "tweet_text_length"
]

# Copy and save metadata
metadata = df[metadata_cols].copy()
metadata.to_csv(os.path.join(output_dir, "metadata.csv"), index=False)

print("Feature extraction and saving completed. Check the 'extracted_features' folder.")