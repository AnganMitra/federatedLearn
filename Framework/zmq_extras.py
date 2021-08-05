from dataclasses import make_dataclass

import zmq

from send_receive_blocking import recv_sub_message


def null_func():
    pass


def null_func_args(*_):
    pass


null_socket = make_dataclass(
    "NullSocket",
    [],
    namespace={
        "close": null_func,
        "send_json": null_func_args,
        "send": null_func_args,
        "send_string": null_func_args,
        "connect": null_func_args,
    },
)


def polling_loop(sockets):
    poller = zmq.Poller()

    for sock in sockets:
        poller.register(sock, zmq.POLLIN)

    while True:
        socks = dict(poller.poll(2000))

        break_loop = False
        for sock in socks:
            if sock in sockets:
                break_loop = break_loop or sockets[sock](sock)

        if break_loop:
            break


def sub_polling_loop(socket, on_message):
    while True:
        events = socket.poll(10000, zmq.POLLIN)
        if events == 0:
            # Gives a chance to break for shutdown events etc
            continue

        topic, message = recv_sub_message(socket)

        if not on_message(topic, message):
            break


def sub_polling_loop_handlers(socket, handlers):
    no_message = handlers.get("no_message", lambda: False)

    while True:
        events = socket.poll(10000, zmq.POLLIN)
        if events == 0:
            # Gives a chance to break for shutdown events etc
            if no_message():
                break
            continue

        topic, message = recv_sub_message(socket)

        handler = handlers.get(topic, lambda _, _1: False)

        if handler(topic, message):
            break
