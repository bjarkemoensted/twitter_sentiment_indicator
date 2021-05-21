import datetime

import numpy as np

import config
import sentiment
import twitter


def update_all():
    twitter.update_economics_tweets()
    sentiment.update_sentiment_data()


def read_twitter_sentiment(min_words=10, earliest_date="2016-01-01", latest_date=None):
    df_twitter = twitter.read_twitter_data()

    # Drop tweets outside desired range
    keepmask = df_twitter["date"] >= earliest_date
    if latest_date:
        keepmask &= df_twitter["date"] <= latest_date
    df_twitter.drop(df_twitter[~keepmask].index, inplace=True)

    df_sentiment = sentiment.read_sentiment_data()

    res = df_twitter.merge(
        right=df_sentiment,
        how="inner",
        on="tweet_id"
    )

    sentiment.add_derived_sentiment_metrics(res)
    too_few_words_mask = res["n_words"] < min_words
    res.drop(res[too_few_words_mask].index, inplace=True)

    # todo this is slow. Can probably be vectorized or something.
    res["datetime"] = [datetime.datetime.strptime(s, config._date_format) for s in res["date"]]
    res.sort_values(by="date", inplace=True)

    return res
