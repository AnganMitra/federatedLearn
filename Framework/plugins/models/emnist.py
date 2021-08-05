def cnn(input_shape=(28, 28, 1), outputs=10):
    import tensorflow.keras.layers as layers
    import tensorflow.keras.models as models
    import tensorflow.keras.losses as losses

    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation="relu", input_shape=input_shape))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation="relu"))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation="relu"))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation="relu"))
    model.add(layers.Dense(outputs))
    model.compile(
        optimizer="adam",
        loss=losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )

    model.get_threadsafe_weights = model.get_weights
    print (model.summary())
    return model


def register(plugin_keys):
    plugin_keys["emnist_cnn"] = cnn


def get_plugin(key):
    if key == "emnist_cnn":
        return cnn

    return None
