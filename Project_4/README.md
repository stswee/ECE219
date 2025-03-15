# Project 4 ECE219

## About
- **Question 1-8**: `ECE219_Project_4_Part_1.ipynb`
- **Question 9**: `ECE219_Project_4_Part_2.ipynb`
- **Question 10**: `Question10` directory

---

## Instructions

### **Question 1-9**
- Change the **filepaths** accordingly before running the notebook.

### **Question 10**
This directory contains scripts for **feature extraction, dataset processing, and regression analysis** on Twitter data.

1. Run **create_dataset.py** to create inital dataset - change the filepaths accordingly
2. Run **correct_and_subsample.py** to remove rows with missing data and subsample
3. Run **feature_extraction.py** to extract TF-IDF and SBERT features along with metadata
4. Run **favorites_regression.py** or **retweet_regression.py** to train LightGBM model for favorite or retweet number prediction (regression)
    - The script outputs RMSE, MAE, R^2 as metrics
    - The script also ouputs top20 feature importance

```bash
Question10/
├── correct_and_subsample.py       # Cleans and subsamples the dataset
├── create_dataset.py              # Processes raw Twitter data into structured format
├── dataset_exploration.ipynb      # Jupyter Notebook for dataset visualization
├── favorites_regression.py        # Predicts number of favorites for a tweet
├── feature_extraction.py          # Extracts TF-IDF, SBERT, and metadata features
├── retweet_regression.py          # Predicts number of retweets for a tweet
