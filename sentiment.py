import afinn
import numpy as np
import os
import pandas as pd
import time

import config
import twitter


def compute_z_scores(vals):
    """Takes an iterable of values, returns corresponding z-scores (no. of standard devs from mean)."""
    mu = np.nanmean(vals)
    std = np.nanstd(vals)
    res = [(val - mu)/std for val in vals]

    return res


def allow_string(fun):
    """Wrapper that makes functions take a first argument as either a string or iterable.
    if string, returns the result from that string. If list of strings, returns list of such results."""

    def wrapped(x, *args, **kwargs):
        if isinstance(x, str):
            lst = [x]
            return fun(lst, *args, **kwargs)[0]
        else:
            return fun(x, *args, **kwargs)
        #
    return wrapped


def make_sentiment_scorer():
    """Returns a sentiment scorer. Use this method everywhere to ensure consistency."""
    clf = afinn.Afinn(language="da")
    return clf


@allow_string
def compute_sentiment(texts, clf=None):
    """Takes a text, or a list of texts, and returns the corresponding sentiment scores."""
    if clf is None:
        clf = make_sentiment_scorer()
    sentiments = [clf.score(s) for s in texts]
    return sentiments


@allow_string
def count_words(texts):
    """Takes a string or list of strings and counts the words."""
    badstarts = ["RT", "http", "#"]
    lens = [
        len([w for w in s.split() if not any(w.startswith(nah) for nah in badstarts)])
        for s in texts]
    return lens


def read_sentiment_data():
    """Reads in existing sentiment data."""
    fn = config.sentiment_data_file
    if not os.path.exists(fn):
        raise FileNotFoundError("No sentiment file found - try running update_sentiment_data()")
    res = pd.read_pickle(fn)
    return res


def update_sentiment_data(update_all = False):
    """Updates sentiment metrics for tweets. If update_all, all tweets stored locally will have their
    sentiment metrics recalculated (use if you've changed the scoring method, for instance).
    Otherwise, only tweets that haven't yet been rated will be computed."""

    # Helper method to print regular progress updates
    _lastprint = time.time()
    def sparseprint(*args, dt=20, **kwargs):
        nonlocal _lastprint
        now = time.time()
        if now - _lastprint >= dt:
            print(*args, **kwargs)
            _lastprint = now

    sentiment_df = None
    try:
        sentiment_df = read_sentiment_data()
    except FileNotFoundError:
        pass

    ids_current = set([]) if sentiment_df is None else set(sentiment_df["tweet_id"])

    # Read in Twitter data
    twitter_df = twitter.read_twitter_data()
    # If not running a complete update, drop the tweets we've already rated
    if not update_all:
        rated_mask = twitter_df["tweet_id"].isin(ids_current)
        twitter_df.drop(twitter_df[rated_mask].index, inplace=True)

    # Stop if there's nothing to do
    if len(twitter_df) == 0:
        print("No new tweets found.")
        return

    # Compute sentiment scores and word counts
    print(f"Assigning sentiment scores to {len(twitter_df)} tweets.")
    sentiment_scores = []
    clf = make_sentiment_scorer()
    for i, s in enumerate(twitter_df["content"]):
        score = compute_sentiment(s, clf=clf)
        sentiment_scores.append(score)
        msg = f"Progress: {(i+1)*100/len(twitter_df):.2g}% ({i+1} tweets)."
        sparseprint(msg)
    print("Done!")

    twitter_df["sentiment"] = sentiment_scores
    twitter_df["n_words"] = count_words(twitter_df["content"])

    # Concatenate with existing results (if any exist)
    sentiment_headers = ["tweet_id", "sentiment", "n_words"]
    new_sentiment_data = twitter_df[sentiment_headers]
    if sentiment_df is None:
        sentiment_df = new_sentiment_data
    else:
        sentiment_df = pd.concat([sentiment_df, new_sentiment_data])

    # Save to disk
    fn = config.sentiment_data_file
    sentiment_df.to_pickle(fn)


def add_derived_sentiment_metrics(df, negative_threshold=-2):
    """Takes a dataframe containing a 'sentiment' column, and adds to it a number of additional metrics,
    such as n_words, adjusted sentiment (sentiment/words), etc."""

    df["adjusted_sentiment"] = [sent/n if n else float('nan') for sent, n in zip(df["sentiment"], df["n_words"])]
    df["isnegative"] = df["sentiment"] < negative_threshold

    df["sentiment_z"] = compute_z_scores(df["sentiment"])
    df["adjusted_sentiment_z"] = compute_z_scores(df["adjusted_sentiment"])

    return


def add_word_occurrences(df, topic2words, inplace=False, prefix="mentions_"):
    """Takes a dictionary mapping a number of 'topics' to words making up that topic.
    An example could be {'corona': ['covid', 'covid-19', 'coronavirus']}.
    For each such topic, adds a boolean column 'mentions_<topic>' to dataframe, denoting whether
    one or more words from that topic are a substring in the 'content' column.
    With the above example, this would add a 'mentions_corona' column, containing True if any of the words
    are contained in the 'content' field, and False otherwise."""

    if inplace:
        res = df
    else:
        res = df.copy()
    for topic, words in topic2words.items():
        header = prefix+topic
        res[header] = [any(w in s.lower() for w in words) for s in res["content"]]

    return None if inplace else res
