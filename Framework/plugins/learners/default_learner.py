from standard_learner import standard_learner


def register(plugin_keys):
    plugin_keys["default_learner"] = standard_learner


def get_plugin(key):
    if key == "default_learner":
        return standard_learner

    return None
