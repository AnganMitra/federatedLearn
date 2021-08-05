import tensorflow as tf


def windowed_dataset(
    series, window_size, prediction_size, batch_size, pred_column=0, shuffle=True
):
    """
    Creates a windowed dataset using tf datasets
    """
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
