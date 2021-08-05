import logging
import zmq

from context import get_context

logger = logging.getLogger("PubSubClient")


def make_pubsub_client(subscriptions=None, include_control=True):
    pub_client = make_pub_client()
    sub_client = make_sub_client(subscriptions, include_control)
    return pub_client, sub_client


def make_sub_client(subscriptions=None, include_control=True):
    ctx = zmq.Context().instance()
    context = get_context()

    subs = subscriptions or []
    if include_control:
        subs.append("control")

    sub_address = context["sub_address"]
    logger.debug("Sub client connecting to %s", sub_address)
    subscriber = ctx.socket(zmq.SUB)
    subscriber.connect(sub_address)

    for sub in subs:
        subscriber.set_string(zmq.SUBSCRIBE, sub)

    return subscriber


def make_pub_client():
    ctx = zmq.Context().instance()
    context = get_context()

    pub_address = context["pub_address"]
    logger.debug("Pub client connecting to %s", pub_address)

    pubber = ctx.socket(zmq.PUB)
    pubber.connect(context["pub_address"])

    return pubber
