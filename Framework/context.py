storage = {"config": {}}


def get_context():
    return storage["config"]


def set_context(initial):
    storage["config"] = initial
