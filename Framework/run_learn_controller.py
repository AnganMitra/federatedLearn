import logging
import signal

import parse_params as pp
from learn_clients import LearningController
from remote_logging import configure_logging


HELP_STRING = "Learning controller"


def get_arg_list():
    parser = pp.create_parser(HELP_STRING)
    pp.add_remote_log(parser)
    pp.add_pubsub_address(parser)

    args = pp.get_args(parser)
    pp.bind_args_to_context(args)

    return args


if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal.SIG_DFL)

    arg_list = get_arg_list()
    configure_logging(arg_list.remote_log)

    client = LearningController()

    logger = logging.getLogger("Controller")

    command = ""

    while command != "q":
        command = input(
            "Enter:\nl to learn\nq to quit\nk to kill servers\na to fed avg\n\n"
        )

        mparts = command.split(" ")
        ctype = mparts[0]

        if ctype == "l":
            client.request_learn()

        if ctype == "la":
            client.request_learn_all()

        if ctype == "k":
            client.kill_workers()

        if ctype == "kk":
            client.kill_pubsub()

        if ctype == "a":
            client.global_averaging()

        if ctype == "st":
            client.wait_global_ready()

        if ctype == "ml":
            if len(mparts) < 2:
                print("Invalid message. Please include a partition list")
                print("")
                continue
            message = [mparts[0]]
            partitions = [int(mp) for mp in mparts[1].split(",")]
            client.request_multi_learn(partitions)

        if ctype == "m":
            if len(mparts) < 2:
                print("Invalid message")
                print("")
                continue
            message = [mparts[1]]
            if len(mparts) > 2:
                message.append(" ".join(mparts[2:]))
            client.raw_message(message)


        print("")

    logger.info("Quitting")
    client.close()
