import signal

import zmq

from learn_servers import single_fedavg_server
import parse_params as pp
from remote_logging import configure_logging

HELP_STRING = "By round learning process"


def build_args():
    parser = pp.create_parser(HELP_STRING)
    pp.add_remote_log(parser)
    pp.add_jid(parser)

    pp.add_storage_provider(parser)
    pp.add_fedavg_provider(parser)
    pp.add_pubsub_bind(parser)
    pp.add_pubsub_address(parser)

    args = pp.get_args(parser)
    pp.bind_args_to_context(args)
    return args


def main():
    signal.signal(signal.SIGINT, signal.SIG_DFL)

    print("Starting byround fed averaging")

    build_args()
    configure_logging(False)

    try:
        server = single_fedavg_server()
        server.average_and_close()
    finally:
        zmq.Context().instance().term()


if __name__ == "__main__":
    main()
