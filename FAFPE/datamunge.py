import numpy as np
from sklearn.preprocessing import MinMaxScaler


def rolling_window(series, window_size):
    """
    Transforms an array of series into an array of sliding window arrays. If
    the passed in series is a matrix, each column will be transformed into an
    array of sliding windows.
    """
    return np.array(
        [
            series[i : (i + window_size)]
            for i in range(0, series.shape[0] - window_size + 1)
        ]
    )


def multi_shuffle(inputs):
    """
    Shuffles arrays of the same length in lockstep.
    """
    indices = np.arange(inputs[0].shape[0])
    np.random.shuffle(indices)

    return list(map(lambda x: x[indices], inputs))


def dual_shuffle(x, y):
    """
    Shuffles two arrays of the same length in the same positions.
    """
    results = multi_shuffle([x, y])
    return results[0], results[1]


def to_timeseries_input(series, lookback, predictions, output_col=0):
    inputs = rolling_window(series[:-predictions, :], lookback)
    outputs = rolling_window(series[lookback:, output_col], predictions)

    return inputs, outputs


def to_timeseries_by_partition(series, partition, lookback, predictions, output_col=0):
    parts = series[partition].unique()

    series_x = []
    series_y = []

    for part in parts:
        ds_partition = (
            series[series[partition] == part].drop([partition], axis=1).values
        )
        ts_x, ts_y = to_timeseries_input(
            ds_partition, lookback, predictions, output_col
        )
        series_x.append(ts_x)
        series_y.append(ts_y)

    return np.concatenate(series_x), np.concatenate(series_y)


def normalize_dataset(df, train_split_index):
    """
    Normalizes the given dataframe, using only the train data. The test train
    split is decided by the passed in split index.

    Args:
        df (dataframe): a pandas dataframe to normalize
        train_split_index (datetime): a datetime object by which to split the
            dataframe
    """
    scaler = MinMaxScaler()
    scaler.fit(df.iloc[0:train_split_index])
    return (scaler, scaler.transform(df))


def rescale(scaler, in_y, orig_columns=5):
    req_shape = np.dstack([in_y for i in range(0, orig_columns)]).reshape(
        in_y.shape[0], orig_columns
    )
    return scaler.inverse_transform(req_shape)[:, 0]


def find_split_index(df, split_date):
    """
    Finds the index of the split point of the dataframe were it to be
    split by the passed in date time.

    Args:
        df (dataframe): a pandas dataframe to split by date
        split_date (datetime): a datetime object by which to split the
            dataframe
    """
    return len(df[df.index < split_date])


def lag_normalize_split(df, split_date, lookback=12, num_predictions=12, output_col=0):
    train_split_index = find_split_index(df, split_date)

    scaler, scaled = normalize_dataset(df, train_split_index)

    train = scaled[0:train_split_index]
    test = scaled[train_split_index:]

    train_x, train_y = to_timeseries_input(
        train, lookback, num_predictions, output_col=output_col
    )
    # train_x, train_y = dual_shuffle(train_x, train_y)

    test_x, test_y = to_timeseries_input(
        test, lookback, num_predictions, output_col=output_col
    )

    return (scaler, train_x, test_x, train_y, test_y)


def remove_last_dim(arr):
    """
    Reshapes the given array to remove the last dimension (this makes
    the assumption that the last dimension is of shape 1).
    """
    return arr.reshape(arr.shape[0], arr.shape[1])


def prepare_inputs_by_partition(
    df,
    partition_col,
    split_date,
    categorical_cols=None,
    output_col=0,
    lookback=12,
    num_predictions=12,
):
    """
    Lags, splits and normalizes a dataframe based around a partition.
    """
    partitions = df[partition_col].unique()
    scalers = {}
    train_x = None
    test_x = None
    train_y = None
    test_y = None
    testset_by_partition = {}

    for partition in partitions:
        df_part = df.loc[df[partition_col] == partition].copy()

        if categorical_cols is None:
            df_cat_train = None
            df_cat_test = None
        else:
            train_split_index = find_split_index(df_part, split_date)
            df_cat_train = df_part.iloc[
                :train_split_index, categorical_cols
            ].values.astype(np.float32)
            df_cat_test = df_part.iloc[
                train_split_index:, categorical_cols
            ].values.astype(np.float32)
            df_part.drop(df_part.columns[categorical_cols], axis=1, inplace=True)

        df_part.drop([partition_col], axis=1, inplace=True)

        scaler, tr_x, te_x, tr_y, te_y = lag_normalize_split(
            df_part,
            split_date,
            output_col=output_col,
            lookback=lookback,
            num_predictions=num_predictions,
        )
        scalers[partition] = scaler

        testset_by_partition[partition] = {
            "test_x": te_x
            if df_cat_test is None
            else [te_x, df_cat_test[0 : len(te_x)]],
            "test_y": remove_last_dim(te_y),
        }

        if train_x is None:
            train_x = tr_x
            test_x = te_x
            train_y = tr_y
            test_y = te_y
            if not df_cat_train is None:
                train_x_cat = df_cat_train[: len(tr_x)]
                test_x_cat = df_cat_test[: len(te_x)]
        else:
            train_x = np.concatenate((train_x, tr_x))
            test_x = np.concatenate((test_x, te_x))
            train_y = np.concatenate((train_y, tr_y))
            test_y = np.concatenate((test_y, te_y))
            if not df_cat_train is None:
                train_x_cat = np.concatenate((train_x_cat, df_cat_train[: len(tr_x)]))
                test_x_cat = np.concatenate((test_x_cat, df_cat_test[: len(te_x)]))

    return (
        scalers,
        train_x if df_cat_train is None else [train_x, train_x_cat],
        test_x if df_cat_test is None else [test_x, test_x_cat],
        remove_last_dim(train_y),
        remove_last_dim(test_y),
        testset_by_partition,
    )


def shuffle_train_sets(train_x, train_y):
    if isinstance(train_x, list):
        shuffled = multi_shuffle([train_x[0], train_x[1], train_y])
        return [shuffled[0], shuffled[1]], shuffled[2]

    return dual_shuffle(train_x, train_y)
