import logging
import os
from multiprocessing import Process
import signal
import time

import zmq

from context import get_context, set_context
from create_base_weights import ensure_base_weights
from learn_clients import LearningController
from local_learner import local_learner
from global_learner import start_all_globallearner_threads
import parse_params as pp
from remote_logging import configure_logging, cleanup_logging
from reporter import start_reporter_thread

HELP_STRING = "Serialized learning"

logger = logging.getLogger("SerializedLearning")


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
    pp.add_partitions(parser)
    pp.add_iterations(parser)

    args = pp.get_args(parser)
    pp.bind_args_to_context(args)
    return args


def ll_process_wrapper(context, partition_id):
    logger = logging.getLogger(f"Partition_{partition_id}")
    logger.info("Starting partition %s", partition_id)

    context["partition_id"] = partition_id
    context["remote_log_proxy"] = False
    set_context(context)
    configure_logging(True)

    local_learner()

    cleanup_logging()

    zmq.Context().instance().destroy(linger=1000)

    print("Process %s finished" % partition_id)


def start_local_learner(partition_id):
    logger.info("Starting process for partition %s", partition_id)
    context = get_context()
    p = Process(target=ll_process_wrapper, args=(context, partition_id,))
    p.start()
    return p


def shutdown(controller, threads):
    controller.kill_workers()
    time.sleep(0.2)
    controller.kill_pubsub()

    logger.debug("Shutting down threads")
    for t in threads:
        if t is None:
            continue
        t.join()
        logger.debug("Thread finished")
    logger.debug("Finished shutting down threads")

    cleanup_logging()

    controller.close()

    zmq.Context().instance().destroy()


def main():
    signal.signal(signal.SIGINT, signal.SIG_DFL)

    print("Starting serialized learning")

    args = build_args()
    configure_logging(True)

    ensure_base_weights()

    gls, _, monitor = start_all_globallearner_threads()
    time.sleep(0.5)
    controller = LearningController()

    rep = start_reporter_thread()
    threads = [rep, gls, monitor]

    partitions = args.partitions

    threads.extend([start_local_learner(i) for i in range(0, partitions)])

    success = controller.collect_connected(partitions)

    if not success:
        logger.error("Unable to start and contact learners")
        shutdown(controller, threads)
        exit(1)

    try:
        for _ in range(args.iterations):
            if not controller.learn_serially():
                logger.error("Error learning serially")
                break
            controller.global_averaging()
            if not controller.wait_globalaveraging():
                logger.error("Error waiting for global averaging")
                break
    finally:
        shutdown(controller, threads)
        os.kill(os.getpid(), signal.SIGTERM)
        exit(0)


if __name__ == "__main__":
    main()
