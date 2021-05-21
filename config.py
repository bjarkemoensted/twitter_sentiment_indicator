import os

_here = os.path.dirname(os.path.realpath(__file__))

data_dir = os.path.join(_here, "data")
plot_dir = os.path.join(_here, "plots")

twitter_data_file = os.path.join(data_dir, 'twitter.p')
sentiment_data_file = os.path.join(data_dir, 'sentiments.p')

_date_format = "%Y-%m-%d"
_month_format = "%Y-%m"

economics_hashtags = [
    "#dkøko",
    "#aktier",
    "#dkbizz",
    "#dkoko",
    "#dkbiz",
    "#arbejde",
    "#arbejdsmarked",
    "#danskøkonomi",
    "#dkfinans",
    "#dkfin",
    "#businessdk"
]


def make_econ_hashtag_query():
    """Returns a Twitter search query for any of the economics-related hashtags"""
    s = " OR ".join(economics_hashtags)
    return s


SCRAPE_EARLIEST_DATE = "2010-01-01"
SCRAPE_LATEST_DATE = None  # None = today
