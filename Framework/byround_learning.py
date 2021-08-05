import signal

import zmq

from learn_servers import single_learner_server
import parse_params as pp
from remote_logging import configure_logging

HELP_STRING = "By round learning process"


def build_args():
    parser = pp.create_parser(HELP_STRING)
    pp.add_remote_log(parser)
    pp.add_epochs(parser)
    pp.add_jid(parser)
    pp.add_pid(parser)
    pp.add_verbose(parser)

    pp.add_data_provider(parser)
    pp.add_model_provider(parser)
    pp.add_report_storage_provider(parser)
    pp.add_storage_provider(parser)
    pp.add_storage_dirs(parser)
    pp.add_fedavg_provider(parser)
    pp.add_pubsub_bind(parser)
    pp.add_pubsub_address(parser)

    args = pp.get_args(parser)
    pp.bind_args_to_context(args)
    return args


def main():
    signal.signal(signal.SIGINT, signal.SIG_DFL)

    print("Starting byround learning")

    build_args()
    configure_logging(False)

    try:
        server = single_learner_server()
        if not server.learn_and_close():
            exit(1)
    finally:
        zmq.Context().instance().term()

    exit(0)


if __name__ == "__main__":
    main()
