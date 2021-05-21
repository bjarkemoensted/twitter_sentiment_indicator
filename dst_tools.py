import datetime
import numpy as np

from dst.dst_utils import to_dataframe
from dst.pydst import get_data

import config
import utils


def parse_time(timestring, default_date=15):
    """Takes a string like 2011M11 and returns datetime objects representing e.g.
    15th November, 2021."""
    year = int(timestring.split("M")[0])
    month = int(timestring.split("M")[1])
    dt = datetime.datetime(year, month, default_date)
    return dt


def get_ms_timestamp(dt):
    """Takes a datetime object and returns the epoch timestamp in milliseconds"""
    ts = (dt - datetime.datetime(1970, 1, 1)).total_seconds() * 1000
    return ts


def parse_consumer_trust(cti):
    """Takes a value for consumer trust (forbrugertillidsindikator) from DST.
    Converts to float - DST uses '..' for missing values for some reason."""
    val = float("nan") if cti == ".." else float(cti)
    return val


def parse_dst_result(df, drop_nans=True, default_date=15):
    """DST returns consumer trust data in a rather idiosyncratic format.
    This parses that into a dataframe with the following columns:
    cti: Consumer trust as a float.
    data: Datetime representation of the month of publication for the cti (defaults to 15th).
    timestamp: Corresponding epoch timestamp in milliseconds."""

    res = df.copy()
    res["cti"] = [parse_consumer_trust(s) for s in res["INDHOLD"]]

    if drop_nans:
        res = res[res["cti"].notnull()]

    res["datetime"] = [parse_time(s, default_date) for s in res["TID"]]
    res["date"] = [datetime.datetime.strftime(dt, config._date_format) for dt in res["datetime"]]
    res["timestamp"] = [get_ms_timestamp(dt) for dt in res["datetime"]]

    del res["INDIKATOR"], res["INDHOLD"], res["TID"]

    return res


def compute_z_scores(vals):
    """Takes an iterable of values, returns corresponding z-scores (no. of standard devs from mean)."""
    mu = np.nanmean(vals)
    std = np.nanstd(vals)
    res = [(val - mu)/std for val in vals]

    return res

def get_consumer_trust(drop_nans=True, default_date=15, earliest_date="2016-01-01", latest_date=None):
    """Downloads consumer trust data from DST."""
    response = get_data(table_id="FORV1", variables={"INDIKATOR": "F1", "Tid": "*"})
    df = to_dataframe(response)
    res = parse_dst_result(df, drop_nans=drop_nans, default_date=default_date)

    # Drop tweets outside desired range
    keepmask = res["date"] >= earliest_date
    if latest_date:
        keepmask &= res["date"] <= latest_date
    res.drop(res[~keepmask].index, inplace=True)

    res["cti_z"] = compute_z_scores(res["cti"])

    return res
