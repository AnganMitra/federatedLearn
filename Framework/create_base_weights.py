import logging
import signal

from context import get_context
import parse_params as pp
from plugins import get_plugin, PluginType
from remote_logging import configure_logging


HELP_STRING = "Create base weights"

logger = logging.getLogger(f"Base weight creator")


def ensure_base_weights():
    context = get_context()

    weight_storage = get_plugin(
        PluginType.weight_store, key=context.get("storage_provider"),
    )

    if not weight_storage:
        logger.error(
            "Weight storage provider plugin %s not found. Quitting",
            context.get("storage_provider"),
        )
        return

    storage = weight_storage()

    if storage.base_weights_exist(context):
        return

    get_model = get_plugin(PluginType.model, context.get("model_provider"))
    if not get_model:
        logger.error(
            "Model provider plugin %s not found. Quitting",
            context.get("model_provider"),
        )
        return

    def get_weights():
        model = get_model()
        return model.get_weights()

    storage.create_base_weights(context, get_weights)


def build_args():
    parser = pp.create_parser(HELP_STRING)
    pp.add_jid(parser)

    pp.add_pubsub_address(parser)
    pp.add_model_provider(parser)

    pp.add_storage_provider(parser)
    pp.add_weight_dir(parser)

    args = pp.get_args(parser)
    pp.bind_args_to_context(args)

    return args


def main():
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    configure_logging(False)
    build_args()

    logging.info("Creating base weights")

    ensure_base_weights()


if __name__ == "__main__":
    main()
