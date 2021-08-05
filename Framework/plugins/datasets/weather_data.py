import datetime
import os

import numpy as np
import pandas as pd

from context import get_context

WEATHER_KEY = "weather"

# logger = logging.getLogger(f"WeatherDataProvider")


def windowed_dataset(
    series, window_size, prediction_size, batch_size, pred_column=0, shuffle=True
):
    """
    Creates a windowed dataset using tf datasets
    """
    import tensorflow as tf

    series = tf.expand_dims(series, axis=-1)
    ds = tf.data.Dataset.from_tensor_slices(series)
    ds = ds.window(window_size + prediction_size, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(window_size + prediction_size + 1))
    if shuffle:
        shuffle_buffer = 1000
        ds = ds.shuffle(shuffle_buffer)
    ds = ds.map(
        lambda w: (
            tf.reshape(w[:window_size], (window_size, series.shape[1])),
            tf.reshape(w[-prediction_size:, pred_column], (prediction_size,)),
        )
    )
    return ds.batch(batch_size).prefetch(1)


class WeatherDataReader:
    def __init__(self):
        import datamunge as dm

        self.split_date = datetime.datetime(2020, 1, 1)

        if os.path.exists("weatherstationdata_red.gzip.csv"):
            file = "weatherstationdata_red.gzip.csv"
        elif os.path.exists("weatherstationdata_red.csv"):
            file = "weatherstationdata_red.csv"
        elif os.path.exists("../data/weatherstationdata_red.csv"):
            file = "../data/weatherstationdata_red.csv"
        elif os.path.exists("./data/weatherstationdata_red.csv"):
            file = "./data/weatherstationdata_red.csv"

        if "gzip" in file:
            df = pd.read_csv(file, compression="gzip", parse_dates=["DateTime"],)
        else:
            df = pd.read_csv(file, parse_dates={"DateTime": ["Date", "HrMn"]},)

        df.Temp = df.Temp.mask(df.Temp > 80)
        df.Temp.fillna(inplace=True, method="ffill")
        df.Dewpt = df.Dewpt.mask(df.Dewpt > 80)
        df.Dewpt.fillna(inplace=True, method="ffill")
        df.RHx = df.RHx.mask(df.RHx > 500)
        df.RHx.fillna(inplace=True, method="ffill")
        df.loc[:, "Name"] = df.Name.str.strip()

        names = df.Name.unique()
        df = df[df.Name != names[-1]]

        df.set_index("DateTime", inplace=True)

        split_index = dm.find_split_index(df, self.split_date)
        means = np.mean(df.iloc[:split_index, 1:])
        stds = np.std(df.iloc[:split_index, 1:])

        df.iloc[:, 1:] = (df.iloc[:, 1:] - means) / stds

        self.names = names[0:-1]
        self.df = df

    def get_number_of_partitions(self):
        return len(self.names)

    def get_partitions(self):
        return self.names

    def get_partition_name(self, pid):
        return self.names[pid]

    def get_data_for_partition(self, partition):
        context = get_context()
        lookback = context.get("wds_lookback", 12)
        n_outputs = context.get("wds_noutputs", 12)
        batch_size = context.get("wds_batchsize", 16)

        partition_name = self.get_partition_name(partition)

        return self.get_windowed_data_for_partition(
            partition_name, lookback, n_outputs, batch_size
        )

    def get_windowed_data_for_partition(
        self, partition, lookback, n_outputs, batch_size
    ):
        import datamunge as dm

        data = self.df[self.df.Name == partition].drop(["Name"], axis=1)
        split_index = dm.find_split_index(data, self.split_date)

        train = data[:split_index].values
        test = data[split_index:].values

        return (
            windowed_dataset(train, lookback, n_outputs, batch_size),
            windowed_dataset(test, lookback, n_outputs, batch_size, shuffle=False),
        )


def get_plugin(key):
    if key == WEATHER_KEY:
        return WeatherDataReader()

    return None
