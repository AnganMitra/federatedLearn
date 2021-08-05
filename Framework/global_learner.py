import json
import logging
import signal
import threading
import time
import zmq

from zmq.devices import ThreadProxySteerable

from common_sockets import (
    connect_thread_worker,
    bind_thread_worker,
)
from context import get_context
from learn_clients import Publisher
from message_constants import BusStatus, MessageTypes, Topics
import parse_params as pp
from plugins import get_plugin, PluginType
from pubsub_client import make_pub_client, make_sub_client
from remote_logging import configure_logging, cleanup_logging
from send_receive_blocking import recv_typed_message_r, send_worker_quit, send_weights
from serialization import deserialize_weights

HELP_STRING = "Global learner process"
TS_ADDRESS = "inproc://global_learner_worker"

logger = logging.getLogger("GlobalLearner")


def close_sockets(sockets):
    for socket in sockets:
        socket.close(linger=3000)


def foward_to_worker_handler(m_type, message, thread_socket, _):
    # message should contain sender, m_data, content
    sender = message[0] if len(message) > 0 else b"nosender"
    print("Forwarding", m_type, sender)
    thread_socket.send(sender, flags=zmq.SNDMORE | zmq.NOBLOCK)
    if len(message) > 1:
        thread_socket.send_string(m_type, flags=zmq.SNDMORE | zmq.NOBLOCK)
        thread_socket.send_multipart(message[1:], flags=zmq.NOBLOCK)
    else:
        thread_socket.send_string(m_type, flags=zmq.NOBLOCK)


def null_handler(_, _1, _2):
    pass


def gl_ping_handler(_, message, _2, pubber):
    sender = message[0].decode()
    print("gl ping", sender)
    pubber.send_multipart([message[0], b"pong"])


def ping_handler(_, _1, _2, pubber):
    print("pong")
    pubber.send_multipart([b"status", b"pong", b"global_learner"])


gl_message_switch = {
    "sw": foward_to_worker_handler,
    "gw": foward_to_worker_handler,
    "avg": foward_to_worker_handler,
    "glping2": foward_to_worker_handler,
    "glping": gl_ping_handler,
    "ping": ping_handler,
}


def load_plugins(context):
    weight_storage = get_plugin(
        PluginType.weight_store, key=context.get("storage_provider"),
    )
    if not weight_storage:
        logger.error(
            "Weight storage provider plugin %s not found. Quitting",
            context.get("storage_provider"),
        )
        return False, None, None

    fedavg_algo = get_plugin(PluginType.fedavg, key=context.get("fedavg_algo"))

    if not fedavg_algo:
        logger.error(
            "Federated averaging algorithm provider plugin %s not found. Quitting",
            context.get("fedavg_algo"),
        )
        return False, None, None

    return True, weight_storage, fedavg_algo


def global_learner_worker():
    thread_socket = connect_thread_worker(TS_ADDRESS)
    pubber = make_pub_client()
    pub = Publisher("globalworker", pubber)
    pub.send_global_worker_status(BusStatus.initializing)

    sockets = [thread_socket, pubber]

    context = get_context()
    success, weight_storage, fedavg_algo = load_plugins(context)
    if not success:
        thread_socket.send(b"f")
        pub.send_global_worker_status(BusStatus.failed)
        thread_socket.close()
        pub.close()
        return

    storage = weight_storage()
    averaging_algo = fedavg_algo()

    try:
        weights = storage.get_base_weights(context)

        if not weights:
            logger.error("Unable to get initial weights in global learner. Quitting")
            raise Exception("No weights found")
    except Exception as e:
        logger.error(e)
        thread_socket.send(b"f")
        pub.send_global_learner_error("Initialization failed")
        pub.send_global_worker_status(BusStatus.failed)
        close_sockets(sockets)
        return

    weight_buffer = []
    thread_socket.send(b"")
    pub.send_global_worker_status(BusStatus.running)

    try:
        while True:
            sender, m_type, contents = recv_typed_message_r(thread_socket)
            if m_type == "Q":
                logger.debug("Global learner worker thread quitting")
                pub.send_global_worker_status(BusStatus.finished)
                break
            if m_type == "gw":
                round_num = int(storage.get_last_round_number()) + 1
                logger.debug(
                    "Received get weights message. Sending round %s", round_num
                )
                send_weights(
                    pubber, weights, round_id=round_num, target=sender, m_type=b"sw"
                )
            if m_type == "avg":
                logger.debug("Received perform averaging")
                last_round = storage.get_latest_weight_round(context)
                if last_round:
                    logger.debug("Performing averaging on %s weights", len(last_round))
                    print("Performing averaging on %s weights" % len(last_round))
                    weights, weight_buffer = averaging_algo(
                        context, weights, last_round
                    )
                    if len(weight_buffer) == 0:
                        metadata = {**context, "wt": "gw"}
                        storage.store_weights(metadata, weights)
                        logger.debug("Weight averaging finished.")
                    logger.debug("Sending averaging done")
                    pub.send_avg_done()
                else:
                    logger.debug("No weights found to average")
                    pub.send_avg_done()
                continue
            if m_type == "glping2":
                pub.send_pong(sender)
                continue
            if m_type == "sw":
                logger.debug("Received weights message")
                m_data = json.loads(contents[0].decode("utf-8"))
                contents = contents[1:]
                weight_buffer.append(deserialize_weights(m_data, contents))

                storage.store_weights({**m_data, **context}, weight_buffer[-1])
                pub.send_weights_stored()

        pub.send_global_worker_status(BusStatus.finished)
    finally:
        close_sockets(sockets)

    logger.debug("Global learner worker thread quit")


def global_learner():
    pubber = make_pub_client()

    worker_socket = bind_thread_worker(TS_ADDRESS)

    sockets = [pubber, worker_socket]

    glw = threading.Thread(target=global_learner_worker)
    glw.start()

    events = worker_socket.poll(20000, zmq.POLLIN)
    if events == 0:
        logger.error("Global worker thread failed to start on time. Quitting")
        pubber.send_multipart([Topics.control.encode(), MessageTypes.kill.encode()])
        time.sleep(0.5)
        close_sockets(sockets)
        return

    msg = worker_socket.recv_string()
    if msg == "f":
        logger.error("Failure initializing worker thread")
        pubber.send_multipart([Topics.control.encode(), MessageTypes.kill.encode()])
        time.sleep(0.5)
        close_sockets(sockets)
        return

    subscriber = make_sub_client(["gl"])
    sockets.append(subscriber)

    try:
        while True:
            parts = subscriber.recv_multipart()
            if len(parts) < 2:
                # Log an error here
                logger.error("Received invalid message")
                continue

            topic = parts[0].decode()
            m_type = parts[1].decode()

            if topic == Topics.control:
                if m_type in (
                    MessageTypes.kill,
                    MessageTypes.all_kill,
                    MessageTypes.global_kill,
                ):
                    logger.info("Global worker kill requested. Quitting")
                    break

            if topic == "gl":
                try:
                    handler = gl_message_switch.get(m_type, null_handler)
                    message = parts[2:] if len(parts) > 2 else []
                    handler(m_type, message, worker_socket, pubber)
                except Exception as e:
                    logger.exception(e)

        send_worker_quit(worker_socket)
    finally:
        close_sockets(sockets)

    glw.join()
    logger.debug("Global worker quit")


def bus_monitor():
    status = BusStatus.initializing

    logger = logging.getLogger("bus_monitor")

    ctx = zmq.Context().instance()

    monitor_address = "inproc://monitor"
    steer_address = "inproc://steerer"

    monitor = ctx.socket(zmq.PULL)
    monitor.connect(monitor_address)
    steerer = ctx.socket(zmq.PAIR)
    steerer.connect(steer_address)

    subscriber = make_sub_client(["gl", "gwstatus", "control", "statusrequest"])
    pubber = make_pub_client()

    sockets = [monitor, steerer, subscriber, pubber]

    poller = zmq.Poller()
    poller.register(monitor, zmq.POLLIN)
    poller.register(subscriber, zmq.POLLIN)

    try:
        while True:
            sockets = dict(poller.poll(2000))
            if monitor in sockets:
                parts = monitor.recv_multipart()
                if len(parts) > 1:
                    logger.debug("Monitor %s", parts[0].decode())
                    continue

            if subscriber in sockets:
                parts = subscriber.recv_multipart()
                topic = parts[0].decode()

                if len(parts) < 2:
                    print(parts[0].decode())
                    continue

                if topic == Topics.control:
                    message = parts[1].decode()
                    if message == MessageTypes.global_kill:
                        logger.info("Monitor global kill received. Terminating")
                        steerer.send_string("TERMINATE")
                        break

                if topic == Topics.global_worker_status:
                    message = parts[1].decode()
                    status = int(message)
                    logger.debug("Setting status %s", status)

                if topic == Topics.status_request:
                    sender = parts[1]
                    logger.debug("Sending status to sender %s", status)
                    pubber.send_multipart([sender, str(status).encode()])
    finally:
        close_sockets(sockets)

    logger.info("Monitor quitting")


def pub_sub_bus():
    context = get_context()
    pub_address, sub_address = context["pub_bind"], context["sub_bind"]
    logger.info("Pub sub bus binding to %s %s", pub_address, sub_address)
    print("Pub sub bus binding to %s %s" % (pub_address, sub_address))

    proxy = ThreadProxySteerable(zmq.XSUB, zmq.XPUB, zmq.PUSH, zmq.PAIR)
    proxy.bind_in(sub_address)
    proxy.bind_out(pub_address)

    proxy.bind_mon("inproc://monitor")
    proxy.bind_ctrl("inproc://steerer")
    proxy.start()

    print("Proxy started")
    return proxy


def start_globallearner_thread():
    gls = threading.Thread(target=global_learner)
    gls.start()
    return gls


def start_pubsub_threads():
    proxy = pub_sub_bus()

    monitor = threading.Thread(target=bus_monitor)
    monitor.start()
    time.sleep(0.1)
    return proxy, monitor


def start_all_globallearner_threads():
    proxy, monitor = start_pubsub_threads()
    gls = start_globallearner_thread()

    return gls, proxy, monitor


def build_args():
    parser = pp.create_parser(HELP_STRING)
    pp.add_remote_log(parser)
    pp.add_jid(parser)

    pp.add_storage_provider(parser)
    pp.add_weight_dir(parser)
    # pp.add_global_learner_addresses(parser)
    pp.add_fedavg_provider(parser)
    pp.add_pubsub_bind(parser)
    pp.add_pubsub_address(parser)

    args = pp.get_args(parser)
    pp.bind_args_to_context(args)

    return args


def main():
    signal.signal(signal.SIGINT, signal.SIG_DFL)

    print("Starting proxy")

    args = build_args()

    configure_logging(args.remote_log)

    gls, proxy, monitor = start_all_globallearner_threads()
    logger.debug("Waiting for global worker to finish")
    gls.join()
    logger.debug("Waiting for monitor to finish")
    monitor.join()

    logger.debug("Waiting for proxy to finish")
    proxy.join()

    cleanup_logging()

    zmq.Context().instance().destroy()

    logger.debug("Global learner all finished. Quitting")


if __name__ == "__main__":
    main()
