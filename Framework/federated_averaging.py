import logging

import numpy as np

logger = logging.getLogger("FederatedAveraging")


class DefaultFederatedAveraging:
    def __init__(self, buflength=5):
        self.buflength = buflength

    def __call__(self, metadata, weights, weight_buffer):
        logger.debug("Federated avg: call with buffer length %s", len(weight_buffer))

        if len(weight_buffer) < self.buflength:
            logger.debug("Federated avg: waiting for more weights, do nothing")
            return weights, weight_buffer

        new_weights = list()
        for weights_list_tuple in zip(*weight_buffer):
            new_weights.append(
                np.array([np.array(w).mean(axis=0) for w in zip(*weights_list_tuple)])
            )

        logger.debug("Federated avg: created new weights. Empty buffer")
        return new_weights, []


def always_average():
    def implementation(_, _1, weight_buffer):
        new_weights = list()
        for weights_list_tuple in zip(*weight_buffer):
            new_weights.append(
                np.array([np.array(w).mean(axis=0) for w in zip(*weights_list_tuple)])
            )

        logger.debug("Federated avg: created new weights. Empty buffer")
        return new_weights, []

    return implementation
