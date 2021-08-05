import logging
import time
from types import SimpleNamespace

import uuid

from learn_clients import LearningController
from local_learner import start_locallearner_thread
from context import get_context
from create_base_weights import ensure_base_weights
from global_learner import start_all_globallearner_threads
from reporter import start_reporter_thread


INPROC_TEMPLATE = "inproc://%s"


def single_fedavg_server():
    logger = logging.getLogger("SingleFedAvgServer")

    logger.info("Starting fedavg server")

    pub_address = INPROC_TEMPLATE % uuid.uuid4()
    sub_address = INPROC_TEMPLATE % uuid.uuid4()
    pub_bind = sub_address
    sub_bind = pub_address

    context = get_context()
    context["pub_address"] = pub_address
    context["sub_address"] = sub_address
    context["pub_bind"] = pub_bind
    context["sub_bind"] = sub_bind

    gls, pub_sub, monitor = start_all_globallearner_threads()
    time.sleep(0.1)

    controller = LearningController()
    time.sleep(0.5)

    threads = [gls, monitor, pub_sub]

    def close():
        logger.warning("Closing all servers")

        controller.kill_workers()

        threads[0].join()

        controller.kill_pubsub()

        threads[1].join()

        controller.close()

    def average_and_close():
        if not controller.wait_global_ready():
            logger.error("Global server not running correctly. Quitting")
            close()
            return
        controller.global_averaging()
        controller.wait_globalaveraging()
        close()

    sn = SimpleNamespace()
    sn.close = close
    sn.server_threads = threads
    sn.global_learner = gls
    sn.controller = controller
    sn.global_average = controller.global_averaging
    sn.average_and_close = average_and_close
    sn.__exit__ = close

    return sn


def single_learner_server():
    logger = logging.getLogger("SingleLearnServer")

    pub_address = INPROC_TEMPLATE % uuid.uuid4()
    sub_address = INPROC_TEMPLATE % uuid.uuid4()
    pub_bind = sub_address
    sub_bind = pub_address

    context = get_context()
    context["pub_address"] = pub_address
    context["sub_address"] = sub_address
    context["pub_bind"] = pub_bind
    context["sub_bind"] = sub_bind

    logger.debug("Starting single learn server")

    ensure_base_weights()

    gls, pub_sub, monitor = start_all_globallearner_threads()
    time.sleep(0.1)

    controller = LearningController()
    reporter = start_reporter_thread()
    locallearner = start_locallearner_thread()

    logger.debug("Single learn server running")

    threads = [reporter, locallearner, gls, monitor, pub_sub]

    def close():
        logger.warning("Closing all servers")
        controller.kill_workers()
        for t in threads[0:2]:
            t.join()

        controller.kill_pubsub()

        for t in threads[2:-1]:
            t.join()

        controller.close()

    def learn_and_close():
        success = True
        if not controller.wait_global_ready():
            logger.error("Global server not running correctly. Quitting")
            success = False

        if not controller.wait_learner_ready():
            logger.error("Unable to contact learn servers. Quitting")
            success = False

        controller.request_learn()
        if not controller.wait_weightsstored():
            logger.error("Did not receive weights stored message")
            success = False

        close()
        return success

    sn = SimpleNamespace()
    sn.close = close
    sn.server_threads = threads
    sn.local_learner = locallearner
    sn.global_learner = gls
    sn.reporter = reporter
    sn.controller = controller
    sn.learn_and_close = learn_and_close
    sn.__exit__ = close

    return sn


def learner_block_server(inproc=False):
    logger = logging.getLogger("LearningBlockServer")

    if inproc:
        pub_address = INPROC_TEMPLATE % uuid.uuid4()
        sub_address = INPROC_TEMPLATE % uuid.uuid4()
        pub_bind = sub_address
        sub_bind = pub_address

        context = get_context()
        context["pub_address"] = pub_address
        context["sub_address"] = sub_address
        context["pub_bind"] = pub_bind
        context["sub_bind"] = sub_bind

    logger.debug("Starting server block for learning")

    ensure_base_weights()

    gls, pub_sub, monitor = start_all_globallearner_threads()
    time.sleep(0.1)

    controller = LearningController()
    reporter = start_reporter_thread()
    locallearner = start_locallearner_thread()

    logger.debug("Learning block server running")

    threads = [locallearner, reporter, gls, monitor, pub_sub]

    def close():
        logger.warning("Closing server block")
        controller.kill_learn_worker_thread()
        locallearner.join()

        controller.kill_workers()
        reporter.join()
        gls.join()

        controller.kill_pubsub()

        for t in threads[2:-1]:
            t.join()

        controller.close()

    def wait_ready():
        success = True
        if not controller.wait_global_ready():
            logger.error("Global server not running correctly. Quitting")
            success = False

        if not controller.wait_learner_ready():
            logger.error("Unable to contact learn servers. Quitting")
            success = False

        return success

    def learn_all(iterations=1):
        for _ in range(iterations):
            controller.request_learn_all()
            controller.global_averaging_via_local()

    def learn_multi(partitions, iterations=1):
        for _ in range(iterations):
            controller.request_multi_learn(partitions)
            controller.global_averaging_via_local()

    def learn_multi_and_close(partitions, iterations=1):
        learn_multi(partitions, iterations)
        close()

    def learn_all_and_close(iterations=1):
        learn_all(iterations)
        close()

    sn = SimpleNamespace()
    sn.close = close
    sn.server_threads = threads
    sn.local_learner = locallearner
    sn.global_learner = gls
    sn.reporter = reporter
    sn.controller = controller

    sn.wait_ready = wait_ready
    sn.learn_all = learn_all
    sn.learn_all_and_close = learn_all_and_close
    sn.learn_multi = learn_multi
    sn.learn_multi_and_close = learn_multi_and_close
    sn.__exit__ = close

    return sn
