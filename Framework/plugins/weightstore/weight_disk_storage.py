import functools
from storage import get_disk_weight_storage

from context import get_context


def register(plugin_keys):
    base_path = get_context().get("weight_dir", "./weights")

    plugin_keys["disk_storage"] = functools.partial(
        get_disk_weight_storage, base_path=base_path
    )


def get_plugin(key):
    base_path = get_context().get("weight_dir", "./weights")

    if key == "disk_storage":
        return functools.partial(get_disk_weight_storage, base_path=base_path)

    return None
