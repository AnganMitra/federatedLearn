import logging
import math
import time

# NUM_EPOCHS = 5
BATCH_SIZE = 20
SHUFFLE_BUFFER = 100
PREFETCH_BUFFER = 10

logger = logging.getLogger(f"EmnistDataProvider")


def preprocess(dataset, shuffle=True):
    import tensorflow as tf

    def batch_format_fn(element):
        """Flatten a batch `pixels` and return the features as an `OrderedDict`."""
        return (
            tf.reshape(element["pixels"], [28, 28, 1]),
            element["label"],
        )

    ds = dataset.map(batch_format_fn)
    if shuffle:
        ds = ds.shuffle(SHUFFLE_BUFFER)

    return ds.batch(BATCH_SIZE).prefetch(PREFETCH_BUFFER)


class EmnistData:
    def __init__(self, client_batch_size=100):
        from tensorflow_federated import simulation as tffs

        attempts = 0
        while attempts < 3:
            try:
                # emnist_train, emnist_test = tffs.datasets.shakespeare.load_data()
                # emnist_train, emnist_test = tffs.datasets.stackoverflow.load_data()
                emnist_train, emnist_test = tffs.datasets.emnist.load_data()
                # emnist_train, emnist_test = tffs.datasets.cifar100.load_data()
                self.emnist_train = emnist_train
                self.emnist_test = emnist_test
                self.client_batch_size = client_batch_size
                break
            except Exception as e:
                if attempts == 3:
                    raise e
                attempts += 1
                logger.error("Failure loading emnist data %s", e)

            time.sleep(0.5)

    def get_number_of_partitions(self):
        return math.floor(len(self.emnist_train.client_ids) / self.client_batch_size)

    def get_client_ids_for_partition(self, partition_no):
        client_start = self.client_batch_size * partition_no
        client_end = client_start + self.client_batch_size
        return (client_start, client_end)

    def get_partition_name(self, pid):
        client_start, client_end = self.get_client_ids_for_partition(pid)
        return f"emnistclients_{client_start}_{client_end}"

    def get_data_for_partition(self, partition_no):
        client_start, client_end = self.get_client_ids_for_partition(partition_no)

        partition_train = self.emnist_train.from_clients_and_fn(
            self.emnist_train.client_ids[client_start:client_end],
            self.emnist_train.create_tf_dataset_for_client,
        ).create_tf_dataset_from_all_clients()

        partition_test = self.emnist_test.from_clients_and_fn(
            self.emnist_test.client_ids[client_start:client_end],
            self.emnist_test.create_tf_dataset_for_client,
        ).create_tf_dataset_from_all_clients()

        train = preprocess(partition_train, shuffle=True)
        test = preprocess(partition_test, shuffle=False)

        return (train, test)


def get_plugin(key):
    if key == "emnist":
        return EmnistData()

    return None
