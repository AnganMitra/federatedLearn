from storage import get_disk_report_writer


def register(plugin_keys):
    plugin_keys["disk_storage"] = get_disk_report_writer


def get_plugin(key):
    if key == "disk_storage":
        return get_disk_report_writer

    return None
