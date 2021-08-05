import json
import logging
import signal
import threading
import uuid

import zmq

from common_sockets import (
    connect_thread_worker,
    bind_thread_worker,
)
from context import get_context
from learn_clients import LearnAlgoClient, Publisher
from message_constants import Topics, MessageTypes
import parse_params as pp
from plugins import get_plugin, PluginType
from pubsub_client import make_pub_client, make_sub_client
from remote_logging import configure_logging, cleanup_logging
from send_receive_blocking import send_training_report_message
from serialization import NumpyEncoder

thread_local = threading.local()

HELP_STRING = "Local learner process"

TS_ADDRESS = "inproc://local_learner_worker_%s"
DS_ADDRESS = "inproc://local_learner_data_%s"

logger = logging.getLogger(f"LocalLearner")


def learn_for_partitions(
    partitions, pid, pub, pubber, data_provider, get_model, learn_algo, data_socket,
):
    for p in partitions:
        logger.debug("Learning for partition %s", p)

        get_data = lambda pt=p: data_provider.get_data_for_partition(pt)
        node_name = data_provider.get_partition_name(p)
        learn_client = LearnAlgoClient(pid, pubber, data_socket, node_name)

        learner = learn_algo(get_model, get_data, learn_client)
        history = learner.on_learn()
        if history:
            logger.debug("Learning finished. Sending report %s", history)
            send_training_report_message(
                pubber, history, flags=0, node_name=node_name, cls=NumpyEncoder
            )
        else:
            pub.send_local_learner_error("Learning process not completed")

    pub.send_learn_finished()


def multi_learn(
    pid, message, pub, pubber, data_provider, get_model, learn_algo, data_socket
):
    partitions = json.loads(message[1].decode())

    learn_for_partitions(
        partitions, pid, pub, pubber, data_provider, get_model, learn_algo, data_socket
    )


def learn_all(pid, pub, pubber, data_provider, get_model, learn_algo, data_socket):
    partitions = range(data_provider.get_number_of_partitions())

    learn_for_partitions(
        partitions, pid, pub, pubber, data_provider, get_model, learn_algo, data_socket
    )


def learner_loop(
    l_context, learn_algo, get_model, data_provider, thread_socket, pubber, data_socket
):
    get_data = lambda: data_provider.get_data_for_partition(context.get("partition_id"))
    context = get_context()
    node_name = None

    partition_id = context.get("partition_id")
    logger.debug("Getting node name for partition id %s", partition_id)
    node_name = data_provider.get_partition_name(partition_id)

    pid = l_context["pid"]
    client = LearnAlgoClient(pid, pubber, data_socket)
    learner = learn_algo(get_model, get_data, client)

    pub = Publisher(pid, pubber)

    # m_data = json.loads(message[0].decode("utf-8"))
    while True:
        events = thread_socket.poll(10000, zmq.POLLIN)
        if events == 0:
            logger.debug("Local learner polling")
            continue

        parts = thread_socket.recv_multipart()
        task = parts[0].decode()

        if task == MessageTypes.worker_quit:
            logger.debug("Learning work loop shutting down")
            break

        elif task == MessageTypes.req_worker_quit:
            thread_socket.send(b"e")
            break

        elif task == MessageTypes.reset_weights:
            logger.info("Resetting weights in local learner")
            learner.on_reset_weights()

        elif task == MessageTypes.req_fedavg:
            logger.info("Requesting fedavg")
            pub.send_global_avg_req()

        elif task == MessageTypes.send_weights:
            logger.info("Trying to send weights")
            learner.on_send_weights()

        elif task == MessageTypes.learn_all:
            learn_all(
                pid, pub, pubber, data_provider, get_model, learn_algo, data_socket
            )

        elif task == MessageTypes.multi_learn:
            multi_learn(
                pid,
                parts,
                pub,
                pubber,
                data_provider,
                get_model,
                learn_algo,
                data_socket,
            )

        elif task == MessageTypes.learn:
            try:
                print("Learn started for", partition_id)
                history = learner.on_learn()

                if history:
                    logger.debug("Learning finished. Sending report %s", history)
                    send_training_report_message(
                        pubber, history, flags=0, node_name=node_name, cls=NumpyEncoder
                    )
                    pub.send_learn_finished()
                else:
                    pub.send_local_learner_error("Learning process not completed")
            except Exception as e:
                logger.exception(e)
                pub.send_local_learner_error(str(e))


def initialise_plugins():
    context = get_context()

    get_model = get_plugin(PluginType.model, context.get("model_provider"))
    if not get_model:
        logger.error(
            "Model provider plugin %s not found. Quitting",
            context.get("model_provider"),
        )
        return False, None, None, None

    data_provider = get_plugin(PluginType.data, context.get("data_provider"))
    if not data_provider:
        logger.error(
            "Data provider plugin %s not found. Quitting", context.get("data_provider")
        )
        return False, None, None, None

    logger.debug("Learn model data plugins loaded")

    learn_algo = get_plugin(
        PluginType.learners, context.get("learn_provider", "default_learner")
    )
    if not learn_algo:
        logger.error(
            "Learn provider plugin %s not found. Quitting",
            context.get("learn_provider", "default_learner"),
        )
        return False, None, None, None

    logger.debug("Learn thread all plugins loaded")

    return True, get_model, data_provider, learn_algo


def learn_thread(l_context):
    logger.debug("Learn thread started")
    pid = l_context["pid"]

    thread_socket = connect_thread_worker(TS_ADDRESS % pid)
    data_socket = connect_thread_worker(DS_ADDRESS % pid)
    pubber = make_pub_client()
    sockets = [thread_socket, data_socket, pubber]

    logger.debug("Learn thread sockets created")

    success, get_model, data_provider, learn_algo = initialise_plugins()

    if not success:
        thread_socket.send(b"f")
        close_sockets(sockets)
        return

    pub = Publisher(pid, pubber)

    logger.debug("Learn thread, initialization finished")
    thread_socket.send(b"")

    try:
        learner_loop(
            l_context,
            learn_algo,
            get_model,
            data_provider,
            thread_socket,
            pubber,
            data_socket,
        )
    except Exception as e:
        logger.exception(e)
        pubber = make_pub_client()
        pub = Publisher(pid, pubber)

        pub.send_local_learner_error(str(e))
    finally:
        close_sockets(sockets)


def quit_handler(_1, _2, worker_socket, _3):
    logger.info("Learner received quit message. Exiting")
    worker_socket.send_string("q")
    return True


def glping_handler(_1, pub, _2, _3):
    logger.debug("Sending global learner ping")
    pub.send_global_learner_ping()
    return False


def glping2_handler(_1, pub, _2, _3):
    logger.debug("Sending global learner ping2")
    pub.send_global_learner_ping2()
    return False


def ping_handler(_1, pub, _2, _3):
    logger.debug("Ping received. Sending pong")
    pub.send_local_learner_pong()
    return False


def passthrough_weights_handler(message, _1, _2, data_socket):
    logger.debug("Sending weights")
    data_socket.send_multipart([b"weights"] + message[1:])
    return False


def passtoworker_handler(message, _2, worker_socket, _3):
    m_type = message[0].decode()
    if len(message) > 1:
        worker_socket.send_string(m_type, zmq.SNDMORE)
        worker_socket.send_multipart(message[1:])
    else:
        worker_socket.send_string(m_type)
    return False


def pong_handler(_1, _2, _3, _4):
    logger.info("pong")
    return False


passthrough_messages = [
    MessageTypes.learn,
    MessageTypes.send_weights,
    MessageTypes.reset_weights,
    MessageTypes.multi_learn,
    MessageTypes.learn_all,
    MessageTypes.req_worker_quit,
    MessageTypes.req_fedavg,
]
passthrough_handlers = {key: passtoworker_handler for key in passthrough_messages}

kill_handlers = {MessageTypes.kill: quit_handler, MessageTypes.quit: quit_handler}
ll_handlers = {
    **kill_handlers,
    **passthrough_handlers,
    MessageTypes.glping: glping2_handler,
    MessageTypes.glping2: glping2_handler,
    MessageTypes.ping: ping_handler,
    MessageTypes.pong: pong_handler,
}
control_handlers = {**kill_handlers}
pid_handlers = {
    **control_handlers,
    **ll_handlers,
    MessageTypes.send_weights: passthrough_weights_handler,
}

topic_handlers = {
    Topics.control: control_handlers,
    Topics.local_learner: ll_handlers,
    Topics.pid: pid_handlers,
}


def on_learn_message(pid, in_topic, message, pub, worker_socket, data_socket):
    if len(message) < 1:
        logger.error("Invalid message received %s", [m.decode() for m in message])
        print("Invalid learn message")
        return False

    topic = Topics.pid if in_topic == pid else in_topic
    m_type = message[0].decode()

    handler = topic_handlers.get(topic, {}).get(m_type)
    if handler is None:
        logger.warning("Unknown message in local learner %s %s", in_topic, m_type)
        return False

    return handler(message, pub, worker_socket, data_socket)


def close_sockets(sockets):
    for socket in sockets:
        socket.close()


def wait_for_global(pid, pub, subscriber):
    logger.debug("Waiting for global server to be available")
    pub.send_global_learner_ping()

    attempts = 0
    while attempts < 60:
        events = subscriber.poll(5000)
        if events == 0:
            logger.info("Waiting for global server")
            attempts += 1
            pub.send_global_learner_ping()
            continue

        msg = subscriber.recv_multipart()
        topic = msg[0].decode()

        if topic == "ll" and msg[1].decode() == "k":
            logger.info("Received shut down waiting for global server")
            return False

        if topic == "ll" and msg[1].decode() == "ping":
            # ignore ping
            continue

        if topic == pid and msg[1].decode() == "pong":
            logger.info("Received pong from global server")
            return True

        logger.info("Learner received unexpected init message %s", topic)

    return False


def local_learner():
    pid = str(uuid.uuid4())
    thread_local.pid = pid
    l_context = {"pid": pid}
    context = get_context()

    partition_id = context.get("partition_id", 0)
    logger = logging.getLogger(f"LL-{partition_id}")

    logger.info("Starting local learner for partition %s", partition_id)
    print("Starting local learner for partition", partition_id, pid)

    pubber = make_pub_client()
    pub = Publisher(pid, pubber)
    subscriber = make_sub_client(["ll", pid])
    worker_socket = bind_thread_worker(TS_ADDRESS % pid)
    data_socket = bind_thread_worker(DS_ADDRESS % pid)
    sockets = [pubber, subscriber, worker_socket, data_socket]

    learning_thread = threading.Thread(
        target=learn_thread, args=(l_context,), daemon=True
    )
    learning_thread.start()

    events = worker_socket.poll(90000, zmq.POLLIN)
    if events == 0:
        logger.error("Local learner worker thread failed to start on time. Quitting")
        close_sockets(sockets)
        return

    msg = worker_socket.recv_string()
    if msg == "f":
        logger.error("Failure initializing worker thread")
        pubber.send_multipart(
            [b"error", pid.encode(), b"learn_failure", b"Init failed"]
        )
        close_sockets(sockets)
        return

    if not wait_for_global(pid, pub, subscriber):
        logger.error("Failure contacting global server")
        worker_socket.send_string("q")
        close_sockets(sockets)
        return

    poller = zmq.Poller()

    poller.register(subscriber, zmq.POLLIN)
    poller.register(worker_socket, zmq.POLLIN)

    try:
        while True:
            sockets = dict(poller.poll(10000))
            if worker_socket in sockets:
                message = worker_socket.recv_string()
                if message == "e":
                    break

            if not learning_thread.is_alive():
                logger.error("No learning thread. Quitting")
                break

            if subscriber in sockets:
                parts = subscriber.recv_multipart()
                topic = parts[0].decode()
                exit_loop = on_learn_message(
                    pid, topic, parts[1:], pub, worker_socket, data_socket
                )
                if exit_loop:
                    break
    finally:
        close_sockets(sockets)

    learning_thread.join()

    logger.info("Local learner has quit")


def start_locallearner_thread():
    llt = threading.Thread(target=local_learner)
    llt.start()

    return llt


def build_args():
    parser = pp.create_parser(HELP_STRING)
    pp.add_remote_log(parser)
    pp.add_epochs(parser)
    pp.add_jid(parser)
    pp.add_pid(parser)
    pp.add_data_provider(parser)
    pp.add_model_provider(parser)
    pp.add_pubsub_address(parser)
    pp.add_verbose(parser)

    args = pp.get_args(parser)
    pp.bind_args_to_context(args)
    return args


def main():
    signal.signal(signal.SIGINT, signal.SIG_DFL)

    print("Starting local learner")

    arg_list = build_args()
    configure_logging(arg_list.remote_log)

    local_learner()

    cleanup_logging()

    zmq.Context().instance().destroy()


if __name__ == "__main__":
    main()
