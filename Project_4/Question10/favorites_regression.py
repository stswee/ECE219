import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

# Load the pre-extracted features and metadata
sbert_feature_path = "extracted_features/sbert_features.npy"
tfidf_feature_path = "extracted_features/tfidf_features.npy"
metadata_path = "extracted_features/metadata.csv"

print("Loading features and metadata...")
sbert_features = np.load(sbert_feature_path)
tfidf_features = np.load(tfidf_feature_path)
metadata = pd.read_csv(metadata_path)

# Specify metadata columns to be used as features
meta_cols = ["num_hastags", "vader_sentiment", 
             "user_followers_count", "user_friends_count", "user_statuses_count"]
meta_features = metadata[meta_cols].values

# Combine TF-IDF features with metadata features
X_tfidf_combined = np.hstack([tfidf_features, meta_features])
print("Combined TF-IDF + metadata features shape:", X_tfidf_combined.shape)

# Similarly, for SBERT features:
X_sbert_combined = np.hstack([sbert_features, meta_features])
print("Combined SBERT + metadata features shape:", X_sbert_combined.shape)

# Create a single train-test split for consistency
y = metadata["favorite_count"].values
y_log = np.log1p(y)  # Using log transformation to handle skewness
indices = np.arange(len(y_log))
train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42)

X_train_tfidf, X_test_tfidf = X_tfidf_combined[train_idx], X_tfidf_combined[test_idx]
X_train_sbert, X_test_sbert = X_sbert_combined[train_idx], X_sbert_combined[test_idx]
y_train, y_test = y_log[train_idx], y_log[test_idx]

print("\nTF-IDF Combined Train shape:", X_train_tfidf.shape, "Test shape:", X_test_tfidf.shape)
print("SBERT Combined Train shape:", X_train_sbert.shape, "Test shape:", X_test_sbert.shape)
print("Target Train shape:", y_train.shape, "Test shape:", y_test.shape)

########################################
# Advanced Model: LightGBM Regression on TF-IDF + Metadata
########################################

print("\nTraining LightGBM Regression on TF-IDF features...")
train_data_tfidf = lgb.Dataset(X_train_tfidf, label=y_train)
val_data_tfidf = lgb.Dataset(X_test_tfidf, label=y_test, reference=train_data_tfidf)

params = {
    'objective': 'regression',
    'metric': 'rmse',
    'learning_rate': 0.05,
    'num_leaves': 64,
    'verbose': -1
}

lgb_model_tfidf = lgb.train(
    params,
    train_data_tfidf,
    num_boost_round=200,
    valid_sets=[val_data_tfidf],
)

print("Training complete for TF-IDF model.")
y_pred_tfidf = lgb_model_tfidf.predict(X_test_tfidf)
print("Predictions on TF-IDF test set complete.")

# Evaluate LightGBM on TF-IDF (RMSE is computed via np.sqrt)
rmse_tfidf = np.sqrt(mean_squared_error(y_test, y_pred_tfidf))
mae_tfidf = mean_absolute_error(y_test, y_pred_tfidf)
r2_tfidf = r2_score(y_test, y_pred_tfidf)

print("Advanced: LightGBM Regression on TF-IDF features")
print(f"RMSE: {rmse_tfidf:.4f}")
print(f"MAE: {mae_tfidf:.4f}")
print(f"R²: {r2_tfidf:.4f}")

########################################
# Advanced Model: LightGBM Regression on SBERT + Metadata
########################################

print("\nTraining LightGBM Regression on SBERT features...")
train_data_sbert = lgb.Dataset(X_train_sbert, label=y_train)
val_data_sbert = lgb.Dataset(X_test_sbert, label=y_test, reference=train_data_sbert)

lgb_model_sbert = lgb.train(
    params,
    train_data_sbert,
    num_boost_round=200,
    valid_sets=[val_data_sbert],
)

print("Training complete for SBERT model.")
y_pred_sbert = lgb_model_sbert.predict(X_test_sbert)
print("Predictions on SBERT test set complete.")

# Evaluate LightGBM on SBERT
rmse_sbert = np.sqrt(mean_squared_error(y_test, y_pred_sbert))
mae_sbert = mean_absolute_error(y_test, y_pred_sbert)
r2_sbert = r2_score(y_test, y_pred_sbert)

print("Advanced: LightGBM Regression on SBERT features")
print(f"RMSE: {rmse_sbert:.4f}")
print(f"MAE: {mae_sbert:.4f}")
print(f"R²: {r2_sbert:.4f}")


# Get feature importance from LightGBM model
importance_tfidf = lgb_model_tfidf.feature_importance(importance_type='gain')
sorted_indices_tfidf = np.argsort(importance_tfidf)[::-1]

vocab_path = "extracted_features/tfidf_vocabulary.csv"
vocab_df = pd.read_csv(vocab_path)
vocab_dict = dict(zip(vocab_df["index"], vocab_df["term"]))  # Convert to dictionary
meta_cols = ["num_hastags", "vader_sentiment", "user_followers_count", "user_friends_count", "user_statuses_count"]

num_tfidf_features = X_tfidf_combined.shape[1] - len(meta_cols)
tfidf_feature_names = [vocab_dict.get(i, f"TFIDF_{i}") for i in range(num_tfidf_features)] + meta_cols

print("\nTop 20 Feature Importances (TF-IDF + Metadata):")
for i in range(20):
    feature_index = sorted_indices_tfidf[i]
    feature_name = tfidf_feature_names[feature_index]  # Map to actual term or metadata
    print(f"Feature '{feature_name}': {importance_tfidf[feature_index]}")

# Get feature importance from LightGBM model
importance_sbert = lgb_model_sbert.feature_importance(importance_type='gain')
sorted_indices_sbert = np.argsort(importance_sbert)[::-1]
num_sbert_features = X_sbert_combined.shape[1] - len(meta_cols)
sbert_feature_names = [f"SBERT_{i}" for i in range(num_sbert_features)] + meta_cols

# Print top 20 most important features for SBERT model
print("\nTop 20 Feature Importances (SBERT + Metadata):")
for i in range(20):
    feature_index = sorted_indices_sbert[i]
    feature_name = sbert_feature_names[feature_index]  # Map to SBERT embedding or metadata
    print(f"Feature '{feature_name}': {importance_sbert[feature_index]}")