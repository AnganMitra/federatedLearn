import logging
import zmq

logger = logging.getLogger(f"CommonSockets")

def connect_reporter(ctx, bind_address):
    socket = ctx.socket(zmq.DEALER)
    socket.connect(bind_address)
    return socket


def bind_reporter(ctx, bind_address):
    socket = ctx.socket(zmq.DEALER)
    socket.bind(bind_address)
    return socket


def connect_learner(ctx, bind_address):
    socket = ctx.socket(zmq.PUSH)
    socket.connect(bind_address)
    return socket


def bind_learner(ctx, bind_address):
    socket = ctx.socket(zmq.PULL)
    socket.bind(bind_address)
    return socket


def connect_global_learner(ctx, bind_address):
    socket = ctx.socket(zmq.DEALER)
    socket.connect(bind_address)
    return socket


def bind_global_learner(ctx, bind_address):
    socket = ctx.socket(zmq.ROUTER)
    socket.bind(bind_address)
    return socket


def connect_thread_worker(bind_address):
    logger.debug("Connecting to thread worker %s", bind_address)
    ctx = zmq.Context().instance()
    worker = ctx.socket(zmq.PAIR)
    worker.connect(bind_address)
    return worker


def bind_thread_worker(bind_address):
    logger.debug("Binding to thread worker %s", bind_address)
    ctx = zmq.Context().instance()
    worker = ctx.socket(zmq.PAIR)
    worker.bind(bind_address)
    return worker
