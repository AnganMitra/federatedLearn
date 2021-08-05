import matplotlib.pyplot as plt
import pandas as pd
import date_utils as dtutils


def history_plot(history, loss_units=None):
    epochs = len(next(iter(history.history.values())))
    ylabel = "loss" if loss_units is None else f"loss ({loss_units})"
    xlabel = "epoch"
    multiline_plot(
        history.history,
        range(1, epochs + 1),
        ticks_from_x=True,
        ylabel=ylabel,
        xlabel=xlabel,
    )


def plot_learning_rate_history(history):
    """
    Uses the history from a model trained with a LearningRateScheduler and plots a chart
    to help pick the optimal learning rate.
    """
    plt.semilogx(history.history["lr"], history.history["loss"])
    plt.axis([1e-8, 1e-4, 0, 30])


def pandas_multiline_plot_from_data(
    data, chart_index=None, figsize=(19, 6),
):
    """
    Takes a passed in dictionary of data names and data, and plots the lines side by side,
    using the pandas plot functionality. The x axis can be configured by passing in a pandas
    index in the chart_index parameter.
    """
    to_plot = pd.DataFrame(data)
    if not chart_index is None:
        data_len = len(next(iter(data.values())))
        to_plot.index = chart_index[0:data_len]

    to_plot.plot(figsize=figsize)


def pandas_multiline_date_plot(
    data, start_date, freq="5min", date_filter=None, figsize=(19, 6),
):
    """
    Takes a passed in dictionary of data names and data, and plots the lines side by side,
    using the pandas plot functionality. The x axis is formed by creating a pandas datetime
    index starting from the parameter start_date. A date filter can be specified to plot
    only a portion of the data by indexing the pandas index (this should be a string or a
    slice).
    """
    to_plot = pd.DataFrame(data)
    data_len = len(next(iter(data.values())))

    date_index = dtutils.make_datetime_index(start_date, freq=freq, periods=data_len)
    to_plot.index = date_index

    if not date_filter is None:
        to_plot = to_plot.loc[date_filter]

    to_plot.plot(figsize=figsize)


def pandas_2line_plot_from_data(
    data1,
    data2,
    data1_name="Data 1",
    data2_name="Data 2",
    chart_index=None,
    figsize=(19, 6),
):
    """
    Takes 2 passed in iterables of data, and plots them side by side, using the pandas
    plot functionality. The x axis can be configured by passing in a pandas index in the
    chart_index parameter.
    """
    data = {data1_name: data1, data2_name: data2}
    pandas_multiline_plot_from_data(data, chart_index=chart_index, figsize=figsize)


def twoline_plot(
    data1,
    data2,
    data1_name="Data 1",
    data2_name="Data 2",
    x_axis=None,
    figsize=(19, 6),
):
    if x_axis is None:
        x = range(len(data1))
        x_min = 0
        x_max = len(x)
    else:
        x = x_axis[0 : len(data1)]
        x_min = min(x)
        x_max = max(x)

    plt.figure(figsize=figsize)
    plt.xlim(x_min, x_max)
    plt.plot(x, data1, label=data1_name)
    plt.plot(x, data2, label=data2_name)
    plt.legend()
    plt.show()


def multiline_plot(
    data,
    x_axis=None,
    figsize=(19, 6),
    ticks_from_x=False,
    xlabel=None,
    ylabel=None,
    title=None,
):
    data_len = len(next(iter(data.values())))
    if x_axis is None:
        x = range(data_len)
        x_min = 0
        x_max = len(x) - 1
    else:
        x = x_axis[0:data_len]
        x_min = min(x)
        x_max = max(x)

    plt.figure(figsize=figsize)
    plt.xlim(x_min, x_max)
    if x_axis is None or ticks_from_x:
        plt.xticks(range(1, data_len))

    for key, value in data.items():
        plt.plot(x, value, label=key)

    if not xlabel is None:
        plt.xlabel(xlabel)
    if not ylabel is None:
        plt.ylabel(ylabel)
    if not title is None:
        plt.title(title)

    plt.legend()
    plt.show()


def multiline_date_plot(
    data, date_start, figsize=(19, 6), freq="5min",
):
    data_len = len(next(iter(data.values())))
    x_axis = dtutils.make_daterange(date_start, periods=data_len, freq=freq)
    multiline_plot(data, x_axis, figsize)
