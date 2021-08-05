import numpy as np
from sklearn.metrics import mean_absolute_error

import datamunge as dm
import plots as plts

ALL_ROOMS = ["babbage", "babyfoot", "jacquard"]


def print_losses(history):
    print("{:<25} {:<10}".format("Loss Name", "Losses"))
    print("")
    for key, item in history.history.items():
        print("{:<25}".format(key), end="")
        for val in item:
            print("{:.5f} ".format(val), end="")
        print("")


def rescaled_mae(scaler, preds, true_vals, orig_columns=5):
    """
    Calculates the MAE after rescaling the data.
    """
    preds_rs = dm.rescale(scaler, preds, orig_columns)
    true_vals_rs = dm.rescale(scaler, true_vals, orig_columns)
    return mean_absolute_error(true_vals_rs, preds_rs)


def print_mae(preds, true_vals):
    """ Prints the MAE for the given prediction and truth values """
    print(f"MAE {mean_absolute_error(true_vals, preds)}")


def print_rescaled_mae(scaler, preds, true_vals, orig_columns=5):
    print(f"MAE rescaled {rescaled_mae(scaler, preds, true_vals, orig_columns)}")


def make_predictions(model, test_x, test_y):
    preds = model.predict(test_x).flatten()
    true_vals = test_y.flatten()
    return preds, true_vals


def run_predictions(model, in_test_x, in_test_y, scaler=None):
    preds, true_vals = make_predictions(model, in_test_x, in_test_y)

    print_mae(preds, true_vals)
    if not scaler is None:
        print_rescaled_mae(scaler, preds, true_vals, in_test_x.shape[2])

    return (preds, true_vals)


def make_bywindow_predictions(model, test_x, test_y):
    if isinstance(test_x, list):
        #lookback = test_x[0].shape[1]
        predictions = test_y.shape[1]
        test_set = [test_x[0][0::predictions], test_x[1][0::predictions]]
    else:
        #lookback = test_x.shape[1]
        predictions = test_y.shape[1]
        test_set = test_x[0::predictions]

    preds = model.predict(test_set).flatten()
    true_vals = test_y[0::predictions].flatten()

    return preds, true_vals


def make_bywindow_multi_predictions(model, test_x, test_sets):
    """
    Using the model and the given test sets, makes predictions based on the window
    used in the train set and joins the predictions together to make one continuous
    prediction. This multi prediction version is used for models that have multiple
    output predictions.
    """
    if isinstance(test_x, list):
        lookback = test_x[0].shape[1]
        test_set = [test_x[0][0::lookback], test_x[1][0::lookback]]
    else:
        lookback = test_x.shape[1]
        test_set = test_x[0::lookback]

    preds = model.predict(test_set)

    preds = [x.flatten() for x in preds]
    truth_sets = [x[0::lookback].flatten() for x in test_sets]

    return preds, truth_sets


def plot_preds_v_truth(preds, true_vals, truth_name="temperature", chart_index=None):
    plts.pandas_2line_plot_from_data(
        true_vals, preds, truth_name, "predictions", chart_index
    )


def chart_index_for_room(df, split_date, room, num_predictions=12):
    test_set = df.loc[df.index >= split_date]
    return test_set[test_set.room == room].iloc[:-num_predictions].index


def prediction_chart_for_room(
    model,
    room,
    room_testsets,
    truth_name="temperature",
    chart_index=None,
    scaler=None,
    orig_columns=5,
):
    in_test_x = room_testsets[room]["test_x"]
    in_test_y = room_testsets[room]["test_y"]

    preds, true_vals = make_bywindow_predictions(model, in_test_x, in_test_y)

    if not scaler is None:
        preds = dm.rescale(scaler, preds, orig_columns)
        true_vals = dm.rescale(scaler, true_vals, orig_columns)

    plot_preds_v_truth(
        preds, true_vals, truth_name=f"{truth_name} {room}", chart_index=chart_index
    )


def prediction_chart_for_partition(
    model,
    partition,
    testsets,
    truth_name="temperature",
    chart_index=None,
    scaler=None,
    orig_columns=5,
):
    in_test_x = testsets[partition]["test_x"]
    in_test_y = testsets[partition]["test_y"]

    preds, true_vals = make_bywindow_predictions(model, in_test_x, in_test_y)

    if not scaler is None:
        preds = dm.rescale(scaler, preds, orig_columns)
        true_vals = dm.rescale(scaler, true_vals, orig_columns)

    plot_preds_v_truth(
        preds,
        true_vals,
        truth_name=f"{truth_name} {partition}",
        chart_index=chart_index,
    )


def prediction_charts_by_room(
    model,
    room_testsets,
    truth_name="temperature",
    chart_index=None,
    rooms=None,
    scalers=None,
    orig_columns=5,
):
    if rooms is None:
        rooms = ALL_ROOMS

    for room in rooms:
        if scalers is None:
            scaler = None
        else:
            scaler = scalers[room]

        prediction_chart_for_room(
            model,
            room,
            room_testsets,
            truth_name=truth_name,
            chart_index=chart_index,
            scaler=scaler,
            orig_columns=orig_columns,
        )


def run_predictions_for_rooms(model, room_testsets, scalers, rooms=None):
    if rooms is None:
        rooms = ALL_ROOMS

    for room in rooms:
        in_test_x = room_testsets[room]["test_x"]
        in_test_y = room_testsets[room]["test_y"]

        scaler = scalers[room]

        print(f"Predictions for {room}")
        run_predictions(model, in_test_x, in_test_y, scaler)
        print("")


def replace_column_in_input(pred_vals, in_vals, col):
    return np.dstack((np.delete(in_vals, col, axis=2), pred_vals))


def get_slice_of_testset(test_x, start_index=0, end_index=1):
    if isinstance(test_x, list):
        return list(map(lambda x: x[start_index:end_index], test_x))

    return test_x[start_index:end_index]


def make_stepthrough_predictions(model, test_x, test_y, output_col=0, window_size=12):
    test_set = get_slice_of_testset(test_x)

    preds = model.predict(test_set).flatten()

    if isinstance(test_x, list):
        test_length = len(test_x[0])
    else:
        test_length = len(test_x)

    for i in range(window_size, test_length, window_size):
        last_prediction = preds[-window_size:]
        input_slice = get_slice_of_testset(test_x, i, i + 1)
        if isinstance(test_x, list):
            new_input = input_slice
            new_input[0] = replace_column_in_input(
                last_prediction, input_slice[0], output_col
            )
        else:
            new_input = replace_column_in_input(
                last_prediction, input_slice, output_col
            )
        preds = np.append(preds, model.predict(new_input).flatten())

    true_vals = test_y.flatten()[0::window_size]
    return preds, true_vals


def stepthrough_predictions_for_daynum(
    model, test_x, test_y, daynum=0, dayindex=0, dayspan=1, output_col=0, window_size=12
):
    slice_start = 288 * daynum + dayindex
    slice_end = slice_start + dayspan * 288
    index_slice = slice(slice_start, slice_end)

    if isinstance(test_x, list):
        test_slice_x = list(map(lambda x: x[index_slice], test_x))
    else:
        test_slice_x = test_x[index_slice]

    test_slice_y = test_y[index_slice]

    return make_stepthrough_predictions(
        model, test_slice_x, test_slice_y, output_col, window_size
    )
