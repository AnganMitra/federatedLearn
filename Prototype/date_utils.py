import pandas as pd


def make_datetime_index(start_date, end_date=None, freq="5min", periods=None):
    return pd.date_range(start=start_date, end=end_date, freq=freq, periods=periods)


def make_daterange(start_date, end_date=None, freq="5min", periods=None):
    return make_datetime_index(
        start_date, end_date, freq, periods=periods
    ).to_pydatetime()
