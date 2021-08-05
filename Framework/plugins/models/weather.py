def basic_1dconv():
    from tensorflow.keras.layers import Conv1D, Dense, Flatten, MaxPooling1D
    from tensorflow.keras.models import Sequential

    activation = "relu"
    input_shape = (12, 4)
    n_outputs = 12

    model = Sequential()
    model.add(
        Conv1D(
            filters=32,
            kernel_size=3,
            activation=activation,
            input_shape=input_shape,
            name="layer1",
        ),
    )
    model.add(Conv1D(filters=32, kernel_size=3, activation=activation, name="layer2"))
    model.add(MaxPooling1D(pool_size=2, name="layer3"))
    model.add(Conv1D(filters=16, kernel_size=3, activation=activation, name="layer4"))
    model.add(MaxPooling1D(pool_size=2, name="layer5"))
    model.add(Flatten(name="layer6"))
    model.add(Dense(100, activation=activation, name="layer7"))
    model.add(Dense(n_outputs, name="layer8"))
    model.compile(loss="mae", optimizer="adam")
    return model


def encoderdecoder():
    from tensorflow.keras.layers import LSTM, Dense, RepeatVector
    from tensorflow.keras.models import Sequential

    loss = "mae"
    activation = "relu"
    input_shape = (12, 4)
    n_outputs = 12

    model = Sequential()
    model.add(LSTM(100, activation=activation, input_shape=input_shape))
    model.add(RepeatVector(n_outputs))
    model.add(LSTM(100, activation=activation, return_sequences=True))
    model.add(Dense(100, activation=activation))
    model.add(Dense(1, activation=activation))
    model.compile(loss=loss, optimizer="adam")
    return model


def simple_gru():
    from tensorflow.keras.layers import GRU, Dense
    from tensorflow.keras.models import Sequential

    loss = "mae"
    optimizer = "adam"
    input_shape = (12, 4)
    n_outputs = 12

    model = Sequential()
    model.add(GRU(32, input_shape=input_shape))
    model.add(Dense(n_outputs))
    model.compile(loss=loss, optimizer=optimizer)
    return model


def get_plugin(key):
    if key == "weather_1dconv":
        return basic_1dconv

    if key == "weather_encoderdecoder":
        return encoderdecoder

    if key == "weather_simplegru":
        return simple_gru

    return None
