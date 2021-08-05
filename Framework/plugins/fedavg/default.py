from federated_averaging import always_average


def register(plugin_keys):
    plugin_keys["always_fedavg"] = always_average


def get_plugin(key):
    if key == "always_fedavg":
        return always_average

    return None
