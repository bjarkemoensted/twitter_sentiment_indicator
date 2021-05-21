import datetime
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from matplotlib import pyplot as plt

import config

# Define Nationalbanken's colors
import sentiment

dec2frac = lambda arr: [v / 255 for v in arr]
tups = [
    (0, 123, 209),
    (146, 34, 156),
    (196, 61, 33),
    (223, 147, 55),
    (176, 210, 71),
    (102, 102, 102),
]
nbcols = tuple(dec2frac(arr) for arr in tups)
nbblue, nbpurple, nbred, nborange, nbgreen, nbgrey = nbcols

# Standard settings for plotting
default_alpha = 1.0
fontsize = 12
legend_fontsize = 12
label_fontdict = dict(size=12)
fontlabelsize = 14


def label_date(ax, datestring, text):
    dt = datetime.datetime.strptime(datestring, config._date_format)
    ax.axvline(x=dt, color="black", linestyle="dotted", label=text)


def set_defaults():
    """Sets matplotlib defaults to look nicer and use Nationalbanken's color scheme."""

    font = {"size": fontsize}

    matplotlib.rc("font", **font)
    # matplotlib.rc("font", family="sans-serif")
    # matplotlib.rc("font", serif="Helvetica Neue")
    plt.rcParams["font.sans-serif"] = "Nationalbank"
    ticksize = 16
    matplotlib.rc("xtick", labelsize=ticksize)
    matplotlib.rc("ytick", labelsize=ticksize)
    matplotlib.rc("axes", labelsize=fontlabelsize)
    matplotlib.rc("figure", figsize=(12, 8))
    matplotlib.rcParams["axes.prop_cycle"] = matplotlib.cycler(color=nbcols)

    matplotlib.rc("legend", fontsize=legend_fontsize)
    matplotlib.rc("legend", frameon=False)


def center_axis(ax, around=0):
    """Ensures an axis is centralized, i.e. its limits go between two values (-c, +c)"""
    a, b = ax.get_ylim()
    if a < around and b > around:
        c = max((abs(val - around) for val in (a, b)))
        ax.set_ylim(around - c, c + around)


def percentify_axis(ax, attr="yaxis", n_significant_digits=2, multiply=100, specifier="g", suffix="%%"):
    """Formats the tickers on the input axis to display as percentages.
    For example, if you do a plot with y values like ".12" you can do
    fig, ax = plt.subplots()
    ax.plot(...)
    percentify_axis(ax, "yaxis")
    ...to get yticks like "12%" instead."""

    def percentify(tick_val, tick_pos):
        p = multiply * tick_val
        s = "%." + str(n_significant_digits) + specifier+suffix
        return s % p

    formatter = matplotlib.ticker.FuncFormatter(percentify)
    getattr(ax, attr).set_major_formatter(formatter)
    return


def rolling_datetime(datetimes, window, function="max"):
    """Takes an iterable of datetime objects or strings representing datetimes, and returns a rolling average
    over them."""

    epochs = pd.Series([dt.timestamp() for dt in datetimes])
    rolling = epochs.rolling(window=window)
    epochs_rolling = getattr(rolling, function)()

    res = [float('nan') if np.isnan(val) else datetime.datetime.fromtimestamp(val) for val in epochs_rolling]
    return res


def plot_negative_fractions(df_sentiment, breakdown, ax=None, window=1000, label_in=None, label_out=None,
                            legend=None):
    """Breaks down tweet data by a boolean column, and plots the fraction of negative tweets for which
    the boolean is false and true, 'out', and 'in', respectively.
    For instance, if the boolean is 'meantions_corona', 'in' is the tweets that do, and 'out' do not."""

    if not ax:
        _, ax = plt.subplots()
    if legend is None:
        legend = any(arg is not None for arg in (label_in, label_out))

    plot_df = df_sentiment.copy()
    plot_df["frac_neg_in"] = [int(b) for b in (plot_df["isnegative"]) & (plot_df[breakdown])]
    plot_df["frac_neg_out"] = [int(b) for b in (plot_df["isnegative"]) & ~(plot_df[breakdown])]

    startind = window - 1
    rolling = plot_df.rolling(window=window)
    datetimes = rolling_datetime(plot_df["datetime"], window=window)[startind:]

    frac_neg_in = rolling["frac_neg_in"].mean()[startind:]
    frac_neg_out = rolling["frac_neg_out"].mean()[startind:]

    ax.fill_between(
        datetimes,
        frac_neg_out,
        color=nbblue,
        label=label_out)

    ax.fill_between(
        datetimes,
        y1=[a + b for a, b in zip(frac_neg_in, frac_neg_out)],
        y2=frac_neg_out,
        color=nbpurple,
        label=label_in)

    if legend:
        ax.legend()

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    percentify_axis(ax)


def plot_sentiment_and_cti(df_sentiment, df_cti, sentiment_window=5000, ax=None, sentiment_label=None,
                           cti_label=None, legend=None):
    """Plots the evolution over time of the sentiment (rolling average), with the consumer trist indicator."""

    if not ax:
        _, ax = plt.subplots()
    if legend is None:
        legend = any(arg is not None for arg in (sentiment_label, cti_label))

    startind = sentiment_window - 1
    sentiment_times = rolling_datetime(df_sentiment["datetime"], window=sentiment_window)[startind:]
    sentiment_data = df_sentiment["adjusted_sentiment_z"]
    sentiments = sentiment_data.rolling(window=sentiment_window).mean()[startind:]

    ax.plot(sentiment_times, sentiments, color=nbblue, label=sentiment_label)

    ax2 = ax.twinx()
    cti_kwargs = dict(color=nbpurple, linestyle="dashed", marker="o")
    cti_data = df_cti["cti"]
    ax2.plot(df_cti["datetime"], cti_data, **cti_kwargs)

    ax.plot([], [], **cti_kwargs, label=cti_label)

    if legend:
        ax.legend()

    mu_cti = cti_data.mean()
    mu_sentiment = sentiment_data.mean()
    center_axis(ax, around=mu_sentiment)
    center_axis(ax2, around=mu_cti)

    return


def save(fig, fn):
    """Saves figure to the plots dir, using fn as filename."""
    path = os.path.join(config.plot_dir, fn)
    fig.savefig(path, bbox_inches="tight")


def plot_covid_impact(df_sentiment, df_cti):
    """This produces the plot fro mthe corona tracker, showing a) rolling avg. of Twitter sentiment z-score vs
    consumer trust, and b) fraction of negative tweets that mention/don't mention corona-related words."""

    set_defaults()
    fig, axes = plt.subplots(ncols=2, figsize=(16, 5))
    longtermax, shorttermax = axes

    day_zero = "2020-02-27"
    dz_label = "Første smittetilfælde i DK"

    ax = longtermax
    plot_sentiment_and_cti(
        df_sentiment,
        df_cti,
        ax=ax,
        sentiment_label="Sentiment for danske økonomi-tweets",
        cti_label="Forbrugertillidsindikator (h.a.)",
        legend=False)

    label_date(ax, day_zero, dz_label)
    ax.set_ylabel("z-score for sentiment")
    ax.legend()

    ##############################
    ax = shorttermax
    recent_data = df_sentiment[df_sentiment["date"] >= "2020-01-01"].copy()
    topic2words = {"corona": ["covid", "corona", "coronavirusdk", "#COVID19dk"]}
    sentiment.add_word_occurrences(recent_data, topic2words, inplace=True)

    plot_negative_fractions(
        recent_data,
        "mentions_corona",
        window=2000,
        ax=ax,
        label_in="Negative tweets – nævner corona",
        label_out="Negative tweets – nævner ikke corona")

    label_date(ax, day_zero, dz_label)
    ax.set_ylabel("Andel af alle økonomi-relaterede tweets")

    fig.tight_layout()
    return fig, axes