import logging

from context import get_context

logger = logging.getLogger(f"LearningAlgo")


class StandardLearner:
    def __init__(self, get_model, get_data, client):
        self.get_model = get_model
        self.model = get_model()
        self.get_data = get_data
        self.client = client

    def on_reset_weights(self):
        self.model = self.get_model()

    def on_send_weights(self):
        self.client.send_weights(self.model.get_weights)

    def train_task(self):
        logger.debug("Learning started")
        context = get_context()
        epochs = context.get("epochs", 1)
        verbose = context.get("verbose", 0)

        train_data, val_data = self.get_data()

        history = self.model.fit(
            train_data, epochs=epochs, verbose=verbose, validation_data=val_data,
        )

        logger.debug("Learning finished. Sending report %s", history.history)
        return history.history

    def on_learn(self):
        attempts = 0
        while attempts < 3:
            try:
                success, _, weights = self.client.get_weights()
                if not success:
                    logger.error(
                        "Unable to fetch weights from global learner. Not performing learning"
                    )
                    continue

                self.model.set_weights(weights)
                history = self.train_task()
                self.client.send_weights(self.model.get_weights())
                return history
            except Exception as e:
                logger.exception("Learn error %s", e)
                attempts += 1
                if attempts == 3:
                    raise e

        return None


def standard_learner(get_model, get_data, client):
    return StandardLearner(get_model, get_data, client)
