import datetime
import os

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import callbacks as kcb
from tensorflow.keras.layers import (
    GRU,
    LSTM,
    Conv1D,
    Dense,
    Flatten,
    MaxPooling1D,
    RepeatVector,
)
from tensorflow.keras.models import Sequential
from tcn import TCN

from tqdm.keras import TqdmCallback
import kerasextras as ke
import plots as plts


def get_tb_logdir():
    return os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))


def learning_rate_finder_train(model, train_x, train_y, loss="mae", epochs=10):
    lr_schedule = keras.callbacks.LearningRateScheduler(
        lambda epoch: 1e-8 * 10 ** (epoch / 20)
    )
    optimizer = tf.keras.optimizers.SGD(lr=1e-8, momentum=0.9)
    model.compile(loss=loss, optimizer=optimizer)
    history = model.fit(
        train_x,
        train_y,
        callbacks=[lr_schedule, TqdmCallback(verbose=1)],
        epochs=epochs,
        verbose=0,
    )
    plts.plot_learning_rate_history(history)
    return history


def train_model(
    model,
    train_x,
    train_y,
    test_x,
    test_y,
    use_tensorboard=False,
    live_valchart=False,
    epochs=10,
    batch_size=16,
    verbose=0,
    early_stopping_patience=4,
):
    callbacks = [TqdmCallback(verbose=1)]

    if early_stopping_patience > 0:
        callbacks.append(
            keras.callbacks.EarlyStopping(patience=4, restore_best_weights=True)
        )

    if use_tensorboard:
        logdir = get_tb_logdir()
        tensorboard_callback = kcb.TensorBoard(
            logdir, histogram_freq=1, profile_batch=0
        )
        callbacks.append(tensorboard_callback)

    if live_valchart:
        plot_losses = ke.PlotLosses()
        callbacks.append(plot_losses)

    return model.fit(
        train_x,
        train_y,
        epochs=epochs,
        batch_size=batch_size,
        verbose=verbose,
        validation_data=(test_x, test_y),
        callbacks=callbacks,
    )


def get_rnn_shapes(x, y):
    return (
        x.shape[1],
        x.shape[2],
        y.shape[1],
    )


def build_model(
    model_func,
    train_x,
    train_y,
    test_x,
    test_y,
    epochs=15,
    batch_size=16,
    use_tensorboard=False,
):
    n_timesteps, n_features, n_outputs = get_rnn_shapes(train_x, train_y)

    verbose = 1
    input_shape = (n_timesteps, n_features)

    # define model
    model = model_func(input_shape, n_outputs)

    out_shape = model.layers[-1].output_shape[1:]
    train_y = train_y.reshape((train_y.shape[0],) + out_shape)
    test_y = test_y.reshape((test_y.shape[0],) + out_shape)

    return train_model(
        model,
        train_x,
        train_y,
        test_x,
        test_y,
        use_tensorboard=use_tensorboard,
        epochs=epochs,
        batch_size=batch_size,
        verbose=verbose,
    )


def model_encoderdecoder(input_shape, n_outputs, activation="relu", loss="mae"):
    model = Sequential()
    model.add(LSTM(100, activation=activation, input_shape=input_shape))
    model.add(RepeatVector(n_outputs))
    model.add(LSTM(100, activation=activation, return_sequences=True))
    model.add(Dense(100, activation=activation))
    model.add(Dense(1, activation=activation))
    model.compile(loss=loss, optimizer="adam")
    return model


def model_encoderdecoder_tanh(input_shape, n_outputs, loss="mae"):
    return model_encoderdecoder(input_shape, n_outputs, activation="tanh", loss=loss)


def model_convnet(input_shape, n_outputs, activation="relu", loss="mae"):
    model = Sequential()
    model.add(
        Conv1D(
            filters=32, kernel_size=3, activation=activation, input_shape=input_shape
        )
    )
    model.add(Conv1D(filters=32, kernel_size=3, activation=activation))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(filters=16, kernel_size=3, activation=activation))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(100, activation=activation))
    model.add(Dense(n_outputs))
    model.compile(loss=loss, optimizer="adam")
    return model


def model_simple_lstm(input_shape, n_outputs, loss="mae"):
    model = Sequential()
    model.add(LSTM(50, input_shape=input_shape))
    model.add(Dense(n_outputs))
    model.compile(loss=loss, optimizer="adam")
    return model


def model_simple_gru(input_shape, n_outputs, loss="mae", optimizer="adam"):
    model = Sequential()
    model.add(GRU(32, input_shape=input_shape))
    model.add(Dense(n_outputs))
    model.compile(loss=loss, optimizer=optimizer)
    return model


def model_simple_tcn(input_shape, n_outputs, loss="mae", optimizer="adam"):
    i = keras.layers.Input(shape=input_shape)

    x = TCN(return_sequences=False)(i)
    x = keras.layers.Dense(n_outputs)(x)

    model = keras.Model(i, x)
    model.compile(optimizer=optimizer, loss=loss)
    return model


def model_gru_conv(input_shape, n_outputs, loss="mae", optimizer="adam"):
    inp = keras.layers.Input(shape=input_shape)
    x = keras.layers.GRU(32, return_sequences=True)(inp)
    x = keras.layers.AveragePooling1D(2)(x)
    x = keras.layers.Conv1D(32, 3, activation="relu", padding="same", name="f_extract")(
        x
    )
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(16, activation="relu")(x)
    out = keras.layers.Dense(n_outputs)(x)

    model = keras.models.Model(inp, out)
    model.compile(optimizer=optimizer, loss=loss)

    return model


def model_conditioned_gru(n_outputs, train_x, loss="mae", optimizer="adam"):
    model = ConditionedGRU(num_outputs=n_outputs)
    # model.call(train_x)
    model.compile(loss=loss, optimizer=optimizer)
    return model


def model_conditioned_lstm(n_outputs, train_x, loss="mae", optimizer="adam"):
    model = ConditionedLSTM(num_outputs=n_outputs)
    model.call(train_x)
    model.compile(loss=loss, optimizer=optimizer)
    return model


class ConditionedLSTM(keras.Model):
    def __init__(self, num_outputs):
        super(ConditionedLSTM, self).__init__()
        self.cond = ConditionalRNNLayer(
            32, cell="LSTM", return_sequences=True, dtype=tf.float32
        )
        self.lstm2 = keras.layers.LSTM(24, return_sequences=False)
        self.out = keras.layers.Dense(num_outputs)

    def call(self, inputs, **kwargs):
        x = self.cond(inputs)
        x = self.lstm2(x)
        x = self.out(x)
        return x


class ConditionedGRU(keras.Model):
    def __init__(self, num_outputs):
        super(ConditionedGRU, self).__init__()
        self.cond = ConditionalRNNLayer(
            32, cell="GRU", return_sequences=False, dtype=tf.float32
        )
        self.out = keras.layers.Dense(num_outputs)

    def call(self, inputs, **kwargs):
        o = self.cond(inputs)
        o = self.out(o)
        return o


class ConditionalRNNLayer(keras.layers.Layer):
    # Arguments to the RNN like return_sequences, return_state...
    def __init__(self, units, cell=keras.layers.LSTMCell, *args, **kwargs):
        """
        Conditional RNN. Conditions time series on categorical data.
        :param units: int, The number of units in the RNN Cell
        :param cell: string, cell class or object (pre-instantiated). In the case of string, 'GRU',
        'LSTM' and 'RNN' are supported.
        :param args: Any parameters of the keras.layers.RNN class, such as return_sequences,
        return_state, stateful, unroll...
        """
        super().__init__()
        self.units = units
        self.final_states = None
        self.init_state = None
        if isinstance(cell, str):
            if cell.upper() == "GRU":
                cell = keras.layers.GRUCell
            elif cell.upper() == "LSTM":
                cell = keras.layers.LSTMCell
            elif cell.upper() == "RNN":
                cell = keras.layers.SimpleRNNCell
            else:
                raise Exception("Only GRU, LSTM and RNN are supported as cells.")
        self._cell = cell if hasattr(cell, "units") else cell(units=units)
        self.rnn = keras.layers.RNN(cell=self._cell, *args, **kwargs)

        # single cond
        self.cond_to_init_state_dense_1 = keras.layers.Dense(units=self.units)

        # multi cond
        max_num_conditions = 10
        self.multi_cond_to_init_state_dense = []
        for _ in range(max_num_conditions):
            self.multi_cond_to_init_state_dense.append(
                keras.layers.Dense(units=self.units)
            )
        self.multi_cond_p = keras.layers.Dense(1, activation=None, use_bias=True)

    def _standardize_condition(self, initial_cond):
        initial_cond_shape = initial_cond.shape
        if len(initial_cond_shape) == 2:
            initial_cond = tf.expand_dims(initial_cond, axis=0)
        first_cond_dim = initial_cond.shape[0]
        if isinstance(self._cell, keras.layers.LSTMCell):
            if first_cond_dim == 1:
                initial_cond = tf.tile(initial_cond, [2, 1, 1])
            elif first_cond_dim != 2:
                raise Exception(
                    "Initial cond should have shape: [2, batch_size, hidden_size]\n"
                    "or [batch_size, hidden_size]. Shapes do not match.",
                    initial_cond_shape,
                )
        elif isinstance(self._cell, keras.layers.GRUCell) or isinstance(
            self._cell, keras.layers.SimpleRNNCell
        ):
            if first_cond_dim != 1:
                raise Exception(
                    "Initial cond should have shape: [1, batch_size, hidden_size]\n"
                    "or [batch_size, hidden_size]. Shapes do not match.",
                    initial_cond_shape,
                )
        else:
            raise Exception("Only GRU, LSTM and RNN are supported as cells.")
        return initial_cond

    def __call__(self, inputs, *args, **kwargs):
        """
        :param inputs: List of n elements:
                    - [0] 3-D Tensor with shape [batch_size, time_steps, input_dim]. The inputs.
                    - [1:] list of tensors with shape [batch_size, cond_dim]. The conditions.
        In the case of a list, the tensors can have a different cond_dim.
        :return: outputs, states or outputs (if return_state=False)
        """
        assert isinstance(inputs, list) and len(inputs) >= 2
        x = inputs[0]
        cond = inputs[1:]
        if len(cond) > 1:  # multiple conditions.
            init_state_list = []
            for ii, c in enumerate(cond):
                init_state_list.append(
                    self.multi_cond_to_init_state_dense[ii](
                        self._standardize_condition(c)
                    )
                )
            multi_cond_state = self.multi_cond_p(tf.stack(init_state_list, axis=-1))
            multi_cond_state = tf.squeeze(multi_cond_state, axis=-1)
            self.init_state = tf.unstack(multi_cond_state, axis=0)
        else:
            cond = self._standardize_condition(cond[0])
            if cond is not None:
                self.init_state = self.cond_to_init_state_dense_1(cond)
                self.init_state = tf.unstack(self.init_state, axis=0)
        out = self.rnn(x, initial_state=self.init_state, *args, **kwargs)
        if self.rnn.return_state:
            outputs, h, c = out
            final_states = tf.stack([h, c])
            return outputs, final_states

        return out
