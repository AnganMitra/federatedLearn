import json
import logging
import zlib

import zmq
from zmq import Socket

from context import get_context
from message_constants import MessageTypes


def send_weights_to_global(
    socket: Socket, weights, sender_name=None, flags=0, copy=True, track=False,
):
    """send a numpy array with metadata"""

    socket.send_string("gl", flags | zmq.SNDMORE)
    socket.send_string("sw", flags | zmq.SNDMORE)
    socket.send_string(sender_name, flags | zmq.SNDMORE)

    send_weights(
        socket,
        weights,
        sender_name=sender_name,
        target=None,
        m_type=None,
        flags=flags,
        copy=copy,
        track=track,
    )


def send_weights(
    socket: Socket,
    weights,
    sender_name=None,
    target=None,
    m_type=None,
    round_id=None,
    flags=0,
    copy=True,
    track=False,
):
    """send a numpy array with metadata"""

    logging.debug("Sending weights")

    dtype = str(weights[0].dtype)
    shapes = [w.shape for w in weights]

    metadata = dict(dtype=dtype, shapes=shapes)
    metadata["type"] = "sw"

    if sender_name:
        metadata["sender"] = sender_name

    if round_id:
        metadata["round_id"] = round_id

    if target:
        logging.debug("Target")
        socket.send(target, flags | zmq.SNDMORE)

    if m_type:
        socket.send(m_type, flags | zmq.SNDMORE)

    socket.send_json(metadata, flags | zmq.SNDMORE)
    for i, w in enumerate(weights):
        send_flags = flags | zmq.SNDMORE if i < len(weights) - 1 else flags
        cw = zlib.compress(w)
        socket.send(cw, send_flags, copy=copy, track=track)


def send_md_json_message(socket: Socket, metadata, message, flags=0):
    """send a json message with metadata"""

    logging.debug("Sending metadata json message")

    socket.send_json(metadata, flags | zmq.SNDMORE)
    socket.send_json(message)


def send_worker_quit(socket: Socket, flags=0):
    """send a quit message to a worker thread"""

    socket.send(b"", flags | zmq.SNDMORE)
    socket.send_string("Q", flags=flags)


def send_get_weights(socket: Socket, flags=0):
    """send a message requesting to get weights"""

    socket.send_string("gw", flags=flags)


def send_training_report_message(
    socket: Socket, report, flags=0, node_name=None, **kwargs
):
    """send a numpy array with metadata"""

    context = get_context()
    name = node_name or context.get("node_name") or context.get("partition_id")

    metadata = {}
    metadata["nn"] = name
    metadata["mt"] = MessageTypes.training_report
    metadata["job_id"] = context.get("job_id", 0)
    metadata["round_id"] = context.get("round_id", 0)

    socket.send_string("lreport", flags | zmq.SNDMORE)
    socket.send_json(metadata, flags | zmq.SNDMORE)
    socket.send_json(report, flags, **kwargs)


def recv_typed_message_r(socket: Socket):
    message = socket.recv_multipart()

    sender = message[0]
    m_type = message[1].decode("utf-8")

    if len(message) < 3:
        return sender, m_type, None

    contents = message[2:] if len(message) > 2 else None
    return sender, m_type, contents


def recv_typed_message(socket: Socket):
    message = socket.recv_multipart()

    m_type = message[0].decode("utf-8")
    contents = message[1:] if len(message) > 1 else None
    return m_type, contents


def recv_typed_md_message(socket: Socket):
    message = socket.recv_multipart()

    m_type = message[0].decode("utf-8")
    if len(message) < 2:
        return m_type, None, None
    m_data = json.loads(message[1].decode("utf-8"))
    contents = message[2:] if len(message) > 2 else None
    return m_type, m_data, contents


def recv_md_json_message(socket: Socket):
    message = socket.recv_multipart()

    m_data = json.loads(message[0].decode("utf-8"))
    message = json.loads(message[1].decode("utf-8"))
    return m_data, message


def recv_sub_message(socket: Socket):
    parts = socket.recv_multipart()
    topic = parts[0].decode()

    message = parts[1:] if len(parts) > 1 else []
    return topic, message
