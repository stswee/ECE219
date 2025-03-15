
import json
import glob
import os
import pandas as pd
import datetime
import pytz
import re


def extract_hashtags(tweet):
    """
    Extract hashtags from the tweet JSON.
    Returns a comma-separated string of hashtags.
    """
    hashtags = tweet.get("tweet", {}).get("entities", {}).get("hashtags", [])
    # Extract the text field from each hashtag and join them with commas
    return ",".join([ht.get("text", "") for ht in hashtags])

def preprocess_tweet(tweet: dict):
    """
    Extracts useful metadata from a tweet JSON object.
    Returns a dictionary of selected features.
    """
    tweet_type = tweet.get("type", None)
    author = tweet['author']['name']
    original_author = tweet['original_author']['name']
    t = tweet.get("tweet", {}) # type: dict
    user = t.get("user", {}) # type: dict

    # Extract tweet-level features
    tweet_text = t.get("text", "")
    tweet_text = re.sub(r'http\S+', '', tweet_text)
    tweet_text = tweet_text.replace("\n", " ")
    unix_time = tweet.get('citation_date')
    if unix_time is None:
        posting_time = None
    else:
        pst_tz = pytz.timezone('America/Los_Angeles')
        posting_time = datetime.datetime.fromtimestamp(unix_time, pst_tz)
    
    retweet_count = tweet.get('metrics', {}).get('citations', {}).get('total', 0)
    favorite_count = t.get("favorite_count", 0)
    
    # Extract hashtags
    hashtags = extract_hashtags(tweet)
    
    # User-level features
    user_followers_count = user.get("followers_count", 0)
    user_friends_count = user.get("friends_count", 0)
    user_statuses_count = user.get("statuses_count", 0)
    
    # Construct dictionary of features
    features = {
        "author": author,
        "original_author": original_author,
        "tweet_type": tweet_type,
        "tweet_text": str(tweet_text),
        "posting_time": posting_time,
        "unix_time": unix_time,
        "retweet_count": retweet_count,
        "favorite_count": favorite_count,
        "hashtags": hashtags,
        "user_followers_count": user_followers_count,
        "user_friends_count": user_friends_count,
        "user_statuses_count": user_statuses_count,
    }
    
    return features

def process_files(file_list:list[str], output_csv="preprocessed_tweets.csv"):
    """
    Process a list of tweet JSON files and save the extracted metadata to a CSV.
    
    Each file is assumed to contain one JSON tweet per line.
    """
    all_data = []
    for file_path in file_list:
        print(f"Processing {file_path} ...")
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    tweet = json.loads(line) #type: dict
                    if tweet['type'] == 'tweet' and tweet['tweet']['in_reply_to_status_id'] == None:
                        if tweet['tweet'].get("text"):
                            features = preprocess_tweet(tweet)
                            features["file_hashtag"] = file_path.split("#")[-1].replace(".txt","")
                            all_data.append(features)
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON in file {file_path}: {e}")
                except Exception as e:
                    print(f"Error processing tweet in file {file_path}: {e}")
    
    # Create a DataFrame and save to CSV
    df = pd.DataFrame(all_data)
    df.to_csv(output_csv, index=False)
    print(f"Saved preprocessed data to {output_csv}")
    return df

if __name__=="__main__":
    filepaths:list[os.PathLike] = glob.glob("/Users/richardlee/Desktop/UCLA/Classes/2025_Winter/ECE219/ECE219/Project_4/tweetdata/*.txt")
    df = process_files(filepaths) # type: pd.DataFrame
    print(len(df))