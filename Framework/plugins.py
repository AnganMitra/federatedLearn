from enum import Enum
import os
import sys


class PluginType(Enum):
    model = 1
    data = 2
    compression = 3
    weight_store = 4
    fedavg = 5
    report_store = 6
    learners = 7


dirnames = {
    PluginType.model: "models",
    PluginType.data: "datasets",
    PluginType.compression: "compression",
    PluginType.weight_store: "weightstore",
    PluginType.fedavg: "fedavg",
    PluginType.report_store: "reportstore",
    PluginType.learners: "learners",
}

plugins = {
    PluginType.model: {},
    PluginType.data: {},
    PluginType.compression: {},
    PluginType.weight_store: {},
    PluginType.fedavg: {},
    PluginType.learners: {},
}

plugin_paths = list(
    {
        os.path.join(os.path.dirname(os.path.realpath(__file__)), "plugins"),
        os.path.join(os.path.realpath(os.curdir), "plugins"),
    }
)


def init_plugin_paths(paths):
    plugin_paths.append(paths)


def init_plugins():
    for key in dirnames:
        dir_name = dirnames[key]
        paths = [os.path.join(p, dir_name) for p in plugin_paths]
        for path in paths:
            if not os.path.exists(path):
                continue
            files = [x[:-3] for x in os.listdir(path) if x.endswith(".py")]
            if not files:
                continue
            sys.path.insert(0, path)

            for plugin in files:
                mod = __import__(plugin)
                if hasattr(mod, "register"):
                    mod.register(plugins[key])


def get_plugin(ptype: PluginType, key: str):
    if ptype not in dirnames:
        return None

    dir_name = dirnames[ptype]
    paths = [os.path.join(p, dir_name) for p in plugin_paths]

    for path in paths:
        if not os.path.exists(path):
            continue
        files = [x[:-3] for x in os.listdir(path) if x.endswith(".py")]
        if not files:
            continue

        sys.path.insert(0, path)
        for file in files:
            mod = __import__(file)
            if hasattr(mod, "get_plugin"):
                plugin = mod.get_plugin(key)
                if plugin:
                    return plugin
    return None


def get_plugins():
    return plugins
