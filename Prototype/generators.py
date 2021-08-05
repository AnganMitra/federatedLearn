import numpy as np


def multi_gen(generators, shuffle=False):
    """
    Generator that combines multiple other generators to return samples. Will
    either return a value from each generator in succession or randomly
    depending on the value of the shuffle parameter.
    """
    i = -1

    while 1:
        i = np.random.randint(0, 3) if shuffle else (i + 1) % len(generators)
        gen = generators[i]
        sample = next(gen)
        yield sample


def ts_generator(
    data,
    lookback,
    target_col=0,
    n_outputs=1,
    step=1,
    min_index=0,
    max_index=None,
    delay=0,
    shuffle=False,
    batch_size=16,
):
    """
    Generator that creates 3d time series shaped data for use in RNN layers
    or similar

    Args:
        data (array): an indexable matrix of timeseries data
        lookback (int): how many timesteps back the input data should go
        delay (int): how many steps into the future the target should be
        min_index (int): point in data at which to start
        max_index (int): point in data at which to finish
        shuffle (boolean): whether to shuffle the samples
        batch_size (int): the number of samples per batch
        step (int): the period in timesteps at which to sample the data
    """
    if max_index is None:
        max_index = len(data) - delay - n_outputs

    i = min_index + lookback

    # if shuffle:
    #    np.random.shuffle(data)

    while 1:
        if shuffle:
            rows = np.random.randint(min_index + lookback, max_index, size=batch_size)
        else:
            if i + batch_size >= max_index:
                i = min_index + lookback
            rows = np.arange(i, min(i + batch_size, max_index))
            i += len(rows)

        samples = np.zeros((len(rows), lookback // step, data.shape[-1]))
        targets = np.zeros((len(rows), n_outputs))

        for j, _ in enumerate(rows):
            indices = range(rows[j] - lookback, rows[j], step)
            samples[j] = data[indices]
            target_start = rows[j] + delay
            target_end = target_start + n_outputs
            targets[j] = data[target_start:target_end, target_col]

        if n_outputs == 1:
            targets = targets.reshape(targets.shape[:-1])

        yield samples, targets, [None]


def ts_seasonal_generator(
    data,
    target_col=0,
    block_size=24,
    n_outputs=12,
    step=1,
    min_index=0,
    max_index=None,
    delay=0,
    shuffle=False,
    batch_size=16,
    freq=5,
):
    """
    Generator that creates 3d time series shaped data for use in RNN layers
    or similar

    Args:
        data (array): an indexable matrix of timeseries data
        lookback (int): how many timesteps back the input data should go
        delay (int): how many steps into the future the target should be
        min_index (int): point in data at which to start
        max_index (int): point in data at which to finish
        shuffle (boolean): whether to shuffle the samples
        batch_size (int): the number of samples per batch
        step (int): the period in timesteps at which to sample the data
    """
    half_sample = block_size // 2
    lookback = (60 // freq) * 24 * 7 + half_sample
    lookback_d = (60 // freq) * 24 + half_sample

    if max_index is None:
        max_index = len(data) - delay - n_outputs

    i = min_index + lookback

    while 1:
        if shuffle:
            rows = np.random.randint(min_index + lookback, max_index, size=batch_size)
        else:
            if i + batch_size >= max_index:
                i = min_index + lookback
            rows = np.arange(i, min(i + batch_size, max_index))
            i += len(rows)

        samples = np.zeros((len(rows), block_size // step, data.shape[-1] * 3))
        targets = np.zeros((len(rows), n_outputs))

        for j, _ in enumerate(rows):
            indices1 = range(rows[j] - block_size, rows[j], step)
            indices2 = range(rows[j] - lookback, rows[j] - lookback + block_size, step)
            indices3 = range(rows[j] - lookback_d, rows[j] - lookback_d + block_size, step)
            data1 = data[indices1]
            data2 = data[indices2]
            data3 = data[indices3]
            all_data = np.hstack((data1, data2, data3))

            #print(samples.shape)
            #print(data1.shape, data2.shape, data3.shape)
            #print(all_data.shape)
            samples[j] = all_data

            target_start = rows[j] + delay
            target_end = target_start + n_outputs
            targets[j] = data[target_start:target_end, target_col]

        if n_outputs == 1:
            targets = targets.reshape(targets.shape[:-1])

        yield samples, targets, [None]
