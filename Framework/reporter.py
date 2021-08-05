import logging
import signal
import threading

from context import get_context
from message_constants import MessageTypes, Topics
import parse_params as pp
from plugins import get_plugin, PluginType
from pubsub_client import make_sub_client
from remote_logging import configure_logging
from serialization import read_report_message, read_control_message
from zmq_extras import sub_polling_loop_handlers

logger = logging.getLogger("Reporter")

HELP_STRING = "Reporter process"


def start_reporter_thread():
    t = threading.Thread(target=reporter,)
    t.start()

    return t


def load_plugins(context):
    report_writer = get_plugin(
        PluginType.report_store, key=context.get("report_storage_provider"),
    )
    if not report_writer:
        logger.error(
            "Report storage provider plugin %s not found. Quitting",
            context.get("report_storage_provider"),
        )
        return False, None

    return True, report_writer


def reporter():
    logger.info("Starting reporter")

    context = get_context()
    ignore_kill = context.get("ignore_kill", False)

    success, report_writer = load_plugins(context)
    if not success:
        return

    writer = report_writer()

    def on_message(_, message):
        m_data, msg = read_report_message(message)

        logger.info("Reporter message received")
        logger.debug("m_data %s", m_data)

        writer({**context, **m_data}, msg)
        return False

    def on_control_message(_, msg):
        message = read_control_message(msg)

        if message in (
            MessageTypes.kill,
            MessageTypes.all_kill,
            MessageTypes.reporter_kill,
            MessageTypes.global_kill,
        ):
            logger.info("Reporter received quit message")
            if not ignore_kill:
                return True

            logger.info("Ignoring kill message")
            return False

        logger.debug("Unknown control message received %s", message)
        return False

    try:
        sub_client = make_sub_client(["lreport"])

        handlers = {
            Topics.control: on_control_message,
            Topics.learning_reports: on_message,
        }
        sub_polling_loop_handlers(sub_client, handlers)
    finally:
        sub_client.close()

    logging.info("Reporting finished")


def build_args():
    parser = pp.create_parser(HELP_STRING)
    pp.add_remote_log(parser)
    pp.add_report_storage_provider(parser)
    pp.add_report_dir(parser)
    pp.add_sub_address(parser)
    pp.add_ignorekill(parser)

    args = pp.get_args(parser)
    pp.bind_args_to_context(args)
    return args


def main():
    signal.signal(signal.SIGINT, signal.SIG_DFL)

    args = build_args()

    configure_logging(args.remote_log)

    reporter()


if __name__ == "__main__":
    main()
