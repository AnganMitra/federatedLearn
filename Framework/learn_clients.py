import json
import logging
import time
import uuid
import zmq

from context import get_context
from message_constants import BusStatus, MessageTypes, Topics
from pubsub_client import make_pub_client, make_sub_client
from send_receive_blocking import send_weights_to_global
from serialization import deserialize_weights


class Publisher:
    def __init__(self, pid, pubber):
        self.pubber = pubber
        self.pid = pid
        self.logger = logging.getLogger(f"Publisher")

    def close(self):
        self.pubber.close()

    def send_get_weights(self):
        self.pubber.send_multipart([b"gl", b"gw", self.pid.encode()])

    def send_get_weights_failed(self, reason="Get weights failed"):
        self.pubber.send_multipart(
            [b"error", self.pid.encode(), b"gwf", reason.encode()]
        )

    def send_global_learner_ping(self):
        self.pubber.send_multipart([b"gl", b"glping", self.pid.encode()])

    def send_global_learner_ping2(self):
        self.pubber.send_multipart([b"gl", b"glping2", self.pid.encode()])

    def send_local_learner_pong(self):
        self.pubber.send_multipart([b"llstatus", b"pong", self.pid.encode()])

    def send_learn_finished(self):
        self.pubber.send_multipart([b"llstatus", b"lfinished", self.pid.encode()])

    def send_local_learner_error(self, e):
        self.pubber.send_multipart([b"error", b"error", self.pid.encode(), e.encode()])

    def send_global_learner_error(self, message):
        self.pubber.send_multipart(
            [b"error", b"error", self.pid.encode(), message.encode()]
        )

    def send_global_worker_status(self, status):
        self.pubber.send_multipart(
            [b"gwstatus", str(status).encode()], flags=zmq.NOBLOCK
        )

    def send_global_avg_req(self):
        self.logger.debug("Send global averaging request")
        self.pubber.send_multipart([b"gl", b"avg"], flags=zmq.NOBLOCK)

    def send_global_avg_req_from_lw(self):
        """ Makes a global averaging request via the a local learner """
        message = [m.encode() for m in [Topics.local_learner, MessageTypes.req_fedavg]]
        self.pubber.send_multipart(message)

    def send_avg_done(self):
        self.pubber.send_multipart([b"glstatus", b"avgdone"])

    def send_pong(self, sender):
        self.pubber.send_multipart(
            [sender, b"pong", self.pid.encode()], flags=zmq.NOBLOCK
        )

    def send_weights_stored(self):
        self.logger.debug("Sending weights stored message")
        self.pubber.send_multipart([b"glstatus", b"weightsstored"])




class LearnAlgoClient:
    """
    Incorporate a new algorithm to the framework.
    """
    def __init__(self, pid, pubber, data_socket, node_name=None):
        self.inner_pubber = pubber
        self.pubber = Publisher(pid, pubber)
        self.data_socket = data_socket
        context = get_context()
        self.name = (
            node_name or context.get("node_name") or str(context.get("partition_id"))
        )
        self.logger = logging.getLogger(f"LearnAlgoClient")

    def send_weights(self, weights):
        send_weights_to_global(self.inner_pubber, weights, sender_name=self.name)

    def get_weights(self):
        self.pubber.send_get_weights()
        events = self.data_socket.poll(10000, zmq.POLLIN)
        if events == 0:
            self.logger.error("Get weights timeout? %s", events)
            self.pubber.send_get_weights_failed("Timeout getting weights")
            return False, None, None

        parts = self.data_socket.recv_multipart()
        if parts[0].decode() != "weights":
            self.logger.error("Unexpected weights message received")
            self.pubber.send_get_weights_failed("Unexpected weights message received")
            return False, None

        m_data = json.loads(parts[1].decode())
        weights = deserialize_weights(m_data, parts[2:])
        if weights is None:
            self.logger.error("Unable to serialize weights")
            self.pubber.send_get_weights_failed("Unable to deserialize weights")
            return False, None, None

        if "round_id" in m_data:
            round_id = m_data["round_id"]
            self.logger.debug("Setting round id in context to %s", round_id)
            get_context()["round_id"] = round_id
        return True, m_data, weights


class LearningController:
    """
    A client that controls learning.
    """

    def __init__(self):
        self.logger = logging.getLogger("LearnServerController")
        self.socket = make_pub_client()
        self.pid = str(uuid.uuid1()).encode()
        self.pubber = Publisher(self.pid, self.socket)
        self.sub_client = make_sub_client(
            [self.pid.decode(), "llstatus", "glstatus", "error", "control"], False
        )
        self.connected = []

    def kill_learn_worker_thread(self):
        """ Sends a message to tell the learn worker thread to shut down when it has
        finished the current job"""

        message = [
            m.encode() for m in [Topics.local_learner, MessageTypes.req_worker_quit]
        ]
        self.socket.send_multipart(message)

    def kill_workers(self):
        """ Sends a message to instruct all connected servers to shut down """
        message = [m.encode() for m in [Topics.control, MessageTypes.kill]]
        self.socket.send_multipart(message)

    def kill_pubsub(self):
        self.socket.send_multipart([b"control", b"gk"])

    def kill_local_learners(self):
        """ Sends a message to instruct all connected servers to shut down """
        self.socket.send_multipart([b"ll", b"k"])

    def global_averaging(self):
        self.logger.debug("Send global averaging request")
        self.socket.send_multipart([b"gl", b"avg"])

    def global_averaging_via_local(self):
        self.pubber.send_global_avg_req_from_lw()

    def collect_connected(self, max_expected=None):
        self.socket.send_multipart([b"ll", b"ping"], flags=zmq.NOBLOCK)
        time.sleep(1)
        attempts = 0
        while attempts < 15:
            events = self.sub_client.poll(5000, zmq.POLLIN)
            if events == 0:
                self.logger.debug("Waiting for learners. Ping")
                self.socket.send_multipart([b"ll", b"ping"], flags=zmq.NOBLOCK)
                attempts = attempts + 1
                continue
            for _ in range(0, events):
                message = self.sub_client.recv_multipart()
                topic = message[0].decode()
                if topic == "error":
                    self.logger.info("Error in learning procss")
                    return False
                if topic == Topics.control:
                    if message[1].decode() in (
                        MessageTypes.kill,
                        MessageTypes.global_kill,
                    ):
                        self.logger.info("Quit command was issued. Stop waiting")
                        return False
                if topic == Topics.local_learner_status:
                    if len(message) > 1 and message[1].decode() == MessageTypes.pong:
                        pid = message[2].decode()
                        self.logger.info("Learner %s is connected", pid)
                        print("Learner is connected", pid)
                        self.connected.append(pid)
                        if max_expected and len(self.connected) >= max_expected:
                            self.logger.debug(
                                "Found all expected learners %s", self.connected
                            )
                            return True
                else:
                    self.logger.debug(
                        "Unknown message received waiting for pong %s", topic
                    )
            attempts = attempts + 1
        return False

    def learn_serially(self):
        self.logger.info("Learn serially for clients %s", self.connected)
        for pid in self.connected:
            self.logger.info("Sending learn message to %s", pid)
            self.socket.send_multipart([pid.encode(), b"l"])
            success = self.wait_weightsstored()
            if not success:
                return False
            self.logger.info("Learn finished for %s %s", pid, success)
        return True

    def wait_global_ready(self):
        attempts = 0
        while attempts < 10:
            self.socket.send_multipart([b"statusrequest", self.pid], flags=zmq.NOBLOCK)
            events = self.sub_client.poll(3000, zmq.POLLIN)
            if events == 0:
                self.logger.debug("Waiting for status request respose.")
                print("Waiting for status request respose.")
                attempts = attempts + 1
                continue
            for _ in range(0, events):
                message = self.sub_client.recv_multipart()
                topic = message[0].decode()
                if topic == "error":
                    self.logger.error(
                        "Error in learning process %s %s",
                        *[p.decode() for p in message[1:]],
                    )
                    return False

                if topic == "control":
                    if message[1].decode() == "gk":
                        self.logger.info("Quit command was issued. Stop waiting")
                        return False

                if topic == self.pid.decode():
                    status = int(message[1].decode())
                    self.logger.debug("Received bus status of %s", status)
                    if status in (
                        BusStatus.failed,
                        BusStatus.shutting_down,
                        BusStatus.finished,
                    ):
                        return False
                    return True

                self.logger.debug("Unknown topic received waiting for ready %s", topic)

            attempts = attempts + 1

        return False

    def wait_learner_ready(self):
        attempts = 0
        self.socket.send_multipart([b"ll", b"ping"], flags=zmq.NOBLOCK)

        while attempts < 30:
            self.logger.debug("Sending ping")
            events = self.sub_client.poll(5000, zmq.POLLIN)
            if events == 0:
                self.logger.debug("Waiting for learners. Ping")
                if attempts % 5 == 0:
                    self.socket.send_multipart([b"ll", b"ping"], flags=zmq.NOBLOCK)
                attempts = attempts + 1

                continue
            for _ in range(0, events):
                message = self.sub_client.recv_multipart()
                topic = message[0].decode()
                if topic == "error":
                    self.logger.info(
                        "Error in learning process %s %s",
                        *[p.decode() for p in message[1:]],
                    )
                    return False
                if topic == "llstatus":
                    if len(message) > 1 and message[1].decode() == "pong":
                        self.logger.info("Received pong from %s", message[2].decode())
                        return True

                self.logger.debug("Unknown message received waiting for pong %s", topic)
            attempts = attempts + 1
        return False

    def wait_weightsstored(self):
        attempts = 0
        while attempts < 15:
            self.logger.debug("Waiting for weights to be stored")

            events = self.sub_client.poll(10000, zmq.POLLIN)
            if events == 0:
                self.logger.debug("Waiting for weights stored")
                attempts = attempts + 1
                continue
            for _ in range(0, events):
                message = self.sub_client.recv_multipart()
                topic = message[0].decode()
                if topic == "error":
                    self.logger.error(
                        "Error in learning process %s %s",
                        *[p.decode() for p in message[-2:]],
                    )
                    return False
                if topic == "glstatus":
                    if len(message) > 1 and message[1].decode() == "weightsstored":
                        self.logger.info("Received weightsstored from global learner")
                        return True

                self.logger.debug(
                    "Unknown message received waiting for glstatus %s", topic
                )

            attempts = attempts + 1
        return False

    def wait_globalaveraging(self, wait_iterations=20):
        attempts = 0

        self.socket.send_multipart([b"statusrequest", self.pid], flags=zmq.NOBLOCK)

        while attempts < wait_iterations:
            self.logger.debug("Waiting for global averaging")

            events = self.sub_client.poll(10000, zmq.POLLIN)
            if events == 0:
                self.logger.debug("Waiting for global averaging")
                attempts = attempts + 1
                continue
            for _ in range(0, events):
                message = self.sub_client.recv_multipart()
                topic = message[0].decode()
                if topic == "error":
                    self.logger.error("Error in global averaging")
                    return False

                if (
                    topic == "glstatus"
                    and len(message) > 1
                    and message[1].decode() == "avgdone"
                ):
                    self.logger.info("Received averaging done message")
                    return True

                if topic == self.pid.decode():
                    status = int(message[1].decode())
                    self.logger.debug("Received bus status of %s", status)
                    if status in (
                        BusStatus.failed,
                        BusStatus.shutting_down,
                        BusStatus.finished,
                    ):
                        self.logger.warning(
                            "Not waiting global averaging as server shutting down"
                        )
                        return False
                    continue

                self.logger.debug(
                    "Unknown message received waiting for glstatus %s", topic
                )

            attempts = attempts + 1
        return False

    def raw_message(self, parts):
        self.logger.info("Sending raw message %s", parts)
        message = list([p.encode() for p in parts])
        self.socket.send_multipart(message, flags=zmq.NOBLOCK)

    def request_learn(self):
        """ Sends a learn message"""
        self.logger.info("Sending LEARN message to clients")
        self.socket.send_multipart([b"ll", b"l"], flag=zmq.NOBLOCK)

    def request_multi_learn(self, partitions):
        """ Sends a learn message"""
        self.logger.info(
            "Sending LEARN message to clients for partitions %s", partitions
        )
        self.socket.send_multipart(
            [b"ll", b"ml", json.dumps(partitions).encode()], flag=zmq.NOBLOCK
        )

    def request_learn_all(self):
        """ Sends a learn message"""
        self.logger.info("Sending LEARN ALL message to clients")
        self.socket.send_multipart([b"ll", b"la"], flag=zmq.NOBLOCK)

    def close(self):
        self.socket.close()
        self.sub_client.close()
