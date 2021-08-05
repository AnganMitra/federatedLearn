"""
Serializing / de-serializing library deals with encoding, decoding.
"""

import json
import logging
import zlib

import numpy as np

logger = logging.getLogger("Serialization")


def deserialize_weights(m_data: dict, content: list):
    """
    return a list, deserializing model weights
    """
    logger.debug("Deserializing weights, length %s", len(content))
    dtype = m_data["dtype"]
    shapes = m_data["shapes"]

    weights = []
    for i, layer in enumerate(content):
        dl = zlib.decompress(layer)
        buf = memoryview(dl)
        data = np.frombuffer(buf, dtype=dtype)
        weights.append(data.reshape(shapes[i]))

    return weights


def read_report_message(message):
    m_data = json.loads(message[0].decode("utf-8"))
    msg = json.loads(message[1].decode("utf-8"))
    return m_data, msg


def read_control_message(message):
    return message[0].decode()


class NumpyEncoder(json.JSONEncoder):
    """ Json encoder for numpy types """

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
