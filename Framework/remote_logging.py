import logging
import threading

import zmq
from zmq.log.handlers import PUBHandler

from context import get_context

logger = logging.getLogger("LoggingConfig")

__log_context__ = {}


class CustomFormatter(logging.Formatter):
    """Logging Formatter to add colors and count warning / errors"""

    grey = "\x1b[38;21m"
    cyan = "\x1b[36;21m"
    yellow = "\x1b[33;21m"
    red = "\x1b[31;21m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    form = "%(name)s | %(asctime)s | %(message)s"
    db_form = "%(name)s | %(asctime)s | %(message)s | (%(filename)s:%(lineno)d)"

    FORMATS = {
        logging.DEBUG: grey + db_form + reset,
        logging.INFO: cyan + form + reset,
        logging.WARNING: yellow + form + reset,
        logging.ERROR: red + db_form + reset,
        logging.CRITICAL: bold_red + db_form + reset,
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


def add_custom_log_handler():
    base_logger = logging.getLogger()
    base_logger.setLevel(logging.DEBUG)

    sh = logging.StreamHandler()
    sh.setLevel(level=logging.DEBUG)
    sh.setFormatter(CustomFormatter())
    base_logger.addHandler(sh)


def set_zmq_log_handler():
    context = get_context()
    connect_address = context.get("remote_log_connect")

    ctx = zmq.Context().instance()
    log_sock = ctx.socket(zmq.PUB)
    log_sock.connect(connect_address)

    zmq_log_handler = PUBHandler(log_sock)
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logger.addHandler(zmq_log_handler)
    logger.socket = log_sock

    __log_context__["log_sock"] = log_sock

    zmq_log_handler.setFormatter(CustomFormatter())

    if hasattr(zmq_log_handler, "setRootTopic"):
        topic = context.get("remote_log_topic", "FedableLogger")
        zmq_log_handler.setRootTopic(topic)

    return logger


def cleanup_logging():
    if "log_sock" in __log_context__:
        __log_context__["log_sock"].close()
    logging.shutdown()


def start_logging_proxy_thread():
    t = threading.Thread(target=start_logging_proxy, daemon=True,)
    t.start()
    return t


def start_logging_proxy():
    ctx = zmq.Context().instance()
    context = get_context()
    bind_address_front = context.get("remote_log_bind_front", "tcp://*8880")
    bind_address_back = context.get("remote_log_bind_back", "tcp://*8881")

    frontend = ctx.socket(zmq.SUB)
    try:
        frontend.bind(bind_address_front)
    except zmq.ZMQError as e:
        logging.warning("Unable to bind remote log proxy. Address in use?")
        logging.error(e)
        return

    frontend.setsockopt(zmq.SUBSCRIBE, b"")

    # Socket facing services
    backend = ctx.socket(zmq.PUB)

    try:
        backend.bind(bind_address_back)
    except zmq.ZMQError as e:
        logging.warning("Unable to bind remote log proxy. Address in use?")
        logging.error(e)
        return

    zmq.proxy(frontend, backend)


def configure_logging(enable_remote):
    if enable_remote:
        context = get_context()
        if context.get("remote_log_proxy", True):
            start_logging_proxy_thread()
        set_zmq_log_handler()

    base_logger = logging.getLogger()
    base_logger.setLevel(logging.DEBUG)

    sh = logging.StreamHandler()
    sh.setLevel(level=logging.DEBUG)
    sh.setFormatter(CustomFormatter())
    base_logger.addHandler(sh)

    logging.getLogger("matplotlib").setLevel(logging.WARNING)

    if enable_remote:
        logger.info("Configuring remote logging")
    else:
        logger.info("Not configuring remote logging")
