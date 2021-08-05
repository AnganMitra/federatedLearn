import logging
import signal

# import zmq

from create_base_weights import ensure_base_weights
from learn_servers import learner_block_server
import parse_params as pp
from remote_logging import configure_logging, cleanup_logging

HELP_STRING = "Multi round learning"

logger = logging.getLogger("Multi round learning")


def build_args():
    parser = pp.create_parser(HELP_STRING)
    pp.add_remote_log(parser)
    pp.add_epochs(parser)
    pp.add_jid(parser)
    pp.add_inproc(parser)
    pp.add_verbose(parser)

    pp.add_data_provider(parser)
    pp.add_model_provider(parser)
    pp.add_report_storage_provider(parser)
    pp.add_storage_provider(parser)
    pp.add_storage_dirs(parser)
    pp.add_fedavg_provider(parser)
    pp.add_pubsub_bind(parser)
    pp.add_pubsub_address(parser)
    pp.add_partitions(parser)
    pp.add_iterations(parser)
    pp.add_learn_partitions(parser)

    args = pp.get_args(parser)
    pp.bind_args_to_context(args)
    return args


def main():
    signal.signal(signal.SIGINT, signal.SIG_DFL)

    print("Starting multi partition learning")

    args = build_args()
    configure_logging(args.remote_log)

    ensure_base_weights()

    try:
        server = learner_block_server(args.inproc)
        if not server.wait_ready():
            exit(1)
        iterations = args.iterations
        if args.parts:
            server.learn_multi_and_close(args.parts, iterations)
        else:
            server.learn_all_and_close(iterations)
    finally:
        cleanup_logging()
        # zmq.Context().instance().destroy()


if __name__ == "__main__":
    main()
