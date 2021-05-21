import pandas as pd
import snscrape.modules
import warnings

import config


class NoSSLVerificationScraper(snscrape.modules.twitter.TwitterSearchScraper):
    """Ugly hack to disable SSL verification and silence warnings from snscrapes Twitter module."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._session.verify = False

    def _request(self, *args, **kwargs):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            return super()._request(*args, **kwargs)


def uninterruptible_pickle(df, filename):
    """Pickle input dataframe to target filename, but catches keyboard interrupts momentarily until saving is,
    completed so no data is lost from killing the script."""

    done = False
    interrupted = False
    while not done:
        try:
            df.to_pickle(filename)
            done = True
        except KeyboardInterrupt:
            print("Saving in progress... Interrupting after save is complete.")
            interrupted = True
        #
    if interrupted:
        print(f"Done saving to {filename}. Re-raising exception.")
        raise KeyboardInterrupt


def parse_tweet(tweet):
    res = {}
    res["date"] = tweet.date.strftime(config._date_format)
    res["month"] = tweet.date.strftime(config._month_format)
    res["tweet_id"] = tweet.id
    res["content"] = tweet.content

    return res


def add_time_window_to_query(base_query, startat, stopat=None):
    """Takes a query such as '"central bank" OR "central banks"' and adds optional flags controlling the time window
    to search in.."""

    parts = [base_query]
    if stopat is not None:
        parts.append("until:" + stopat)
    parts.append("since:" + startat)
    query = " ".join(parts)
    return query


def build_query_from_search_terms(search_terms, startat, stopat=None, danish_only=False):
    """Builds a query from list of search terms, such as ['money', 'finance', 'etc...']."""
    if isinstance(search_terms, str):
        search_terms = [search_terms]

    term_string = " OR ".join(search_terms)
    base_query = "(%s)" % term_string
    if danish_only:
        base_query += " lang:da"

    query = add_time_window_to_query(base_query=base_query, startat=startat, stopat=stopat)
    return query


def scrape_between(
        base_query,
        callback,
        startat,
        stopat=None,
        buffer_length=100,
        max_tweets=None,
        verbose=True):
    """Takes a Twitter query string containing everything except start and stop dates, and scrapes tweets matching
    the query between the input time delimiters.
    Every time buffer_length tweets have been scraped, the tweets are sent to the callback method."""

    if any(bad in base_query for bad in ("until:", "since:")):
        raise ValueError("base query must not contain temporal info.")

    if max_tweets is None:
        max_tweets = float('inf')

    query = add_time_window_to_query(base_query=base_query, startat=startat, stopat=stopat)
    scraper = NoSSLVerificationScraper(query=query)

    tweets = []
    n_hits = 0
    for tweet in scraper.get_items():
        tweets.append(tweet)
        n_hits += 1
        if n_hits >= max_tweets:
            callback(tweets)
            print(f"Exceeded max of {max_tweets} tweets!")
            return True

        if len(tweets) >= buffer_length:
            callback(tweets)
            tweets = []
            if verbose:
                print("Processed %d tweets!" % buffer_length)
        #

    if tweets:
        callback(tweets)
    if verbose:
        print("Done scraping!")
    return True


def read_twitter_data():
    """Reads in any data that's already been downloaded. Returns None if none is found."""
    res = None
    try:
        res = pd.read_pickle(config.twitter_data_file)
    except FileNotFoundError:
        pass
    return res


def determine_missing_periods(data, earliest=None, latest=None):
    """Determines the periods in which we need to look for new data."""

    if earliest is None:
        earliest = config.SCRAPE_EARLIEST_DATE
    if latest is None:
        latest = config.SCRAPE_LATEST_DATE

    periods = []
    if data is None:
        periods = [(earliest, latest)]
    else:
        a_data = min(data["date"])
        b_data = max(data["date"])
        missing_early_data = earliest != a_data
        periods.append((b_data, None))
        if missing_early_data:
            periods.append((earliest, a_data))
    return periods


def update_economics_tweets(query=None, target_filename=None, save_every=5000):

    if query is None:
        query = config.make_econ_hashtag_query()

    if target_filename is None:
        target_filename = config.twitter_data_file

    print(f"Updating data using query: {query}.\nGoing to save to {target_filename}.")
    running = read_twitter_data()

    periods = determine_missing_periods(running)
    print(f"Querying {len(periods)} periods: {str(periods)}.")

    def update_data_and_save(new_tweets):
        nonlocal running
        parsed = [parse_tweet(tweet) for tweet in new_tweets]
        reached = parsed[-1]["date"]
        new_df = pd.DataFrame(data=parsed)
        if running is None:
            running = new_df
        else:
            running = running.append(new_df, ignore_index=True)

        n_tweets_with_duplicates = len(running)
        running.drop_duplicates(subset="tweet_id", inplace=True)
        n_tweets = len(running)
        dropped = n_tweets_with_duplicates - n_tweets
        if dropped:
            print(f"Dropped {dropped} duplicate Tweet IDs.")

        msg = f"Reached date: {reached}. Downloaded {n_tweets} tweets total."
        print(msg)

        uninterruptible_pickle(df=running, filename=target_filename)

    for startat, stopat in periods:
        print(f"Querying period from {startat} to {'now' if stopat is None else stopat}.")
        scrape_between(
            base_query=query,
            startat=startat,
            stopat=stopat,
            callback=update_data_and_save,
            buffer_length=save_every,
            verbose=False)
        print("Done.")

    return


if __name__ == '__main__':
    pass
