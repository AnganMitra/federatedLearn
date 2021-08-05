import bz2
import functools
import glob
import json
import logging
import os
from pathlib import Path
import pickle
import sys
from itertools import takewhile
from types import SimpleNamespace

import numpy as np

import _pickle as cPickle

from context import get_context


def get_round_from_name(fname: str):
    parts = fname.split("_")
    return parts[1]


def round_equals(fname, round_no):
    round_in_name = get_round_from_name(fname)
    return round_in_name == round_no


def max_round(files):
    return max([get_round_from_name(file) for file in files])


def get_last_globalweight_file(job_no, base_path="./weights"):
    search = os.path.join(base_path, f"{job_no}_*gw.bz2")
    files = glob.glob(search)
    file_names = [os.path.basename(file) for file in files]
    ordered = sorted(file_names, key=get_round_from_name, reverse=True)
    return next(iter(ordered), None)


def get_last_localweight_files(job_no, base_path="./weights"):
    search = os.path.join(base_path, f"{job_no}_*lw.bz2")
    files = glob.glob(search)
    file_names = [os.path.basename(file) for file in files]
    ordered = sorted(file_names, key=get_round_from_name, reverse=True)
    first_item = next(iter(ordered), None)
    if first_item is None:
        return None
    required_round = get_round_from_name(first_item)
    round_check = functools.partial(round_equals, round_no=required_round)
    return list(takewhile(round_check, ordered))


def get_last_partition_localweight_file(job_no, partition, base_path="./weights"):
    search = os.path.join(base_path, f"{job_no}_*{partition}_lw.bz2")
    files = glob.glob(search)
    file_names = [os.path.basename(file) for file in files]
    ordered = sorted(file_names, key=get_round_from_name, reverse=True)
    first_item = next(iter(ordered), None)
    if first_item is None:
        return None
    required_round = get_round_from_name(first_item)
    round_check = functools.partial(round_equals, round_no=required_round)
    return list(takewhile(round_check, ordered))


def load_weights_from_file(file_path):
    with bz2.BZ2File(file_path, "r") as file:
        npzfile = np.load(file)
        return list([npzfile[name] for name in npzfile.files])


def get_null_weight_storage():
    def null_func(*_):
        return None

    sn = SimpleNamespace()
    sn.get_base_weights = null_func
    sn.store_weights = null_func
    sn.get_latest_weight_round = null_func
    sn.ensure_default_weights = null_func

    return sn


def get_disk_report_writer():
    context = get_context()
    base_path = context.get("report_dir", "./reports")

    Path(base_path).mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("ReporterDiskStorage")
    base_store = get_disk_storage()

    def store_report(metadata, report):
        job_id = metadata.get("job_id", 0)
        round_id = metadata.get("round_id", 0)
        sender = metadata.get("nn", "unknown")

        logger.info(
            "Storing training report from %s, job, round %s %s",
            sender,
            job_id,
            round_id,
        )
        logger.debug("Metadata %s", metadata)
        logger.debug(report)

        file_name = f"{job_id}_{round_id}_{sender}_report.json"
        path = os.path.join(base_path, file_name)
        try:
            base_store.store_json(path, report)
        except OSError:
            logging.error("Error storing report %s", sys.exc_info())

    return store_report


def get_default_weight_storage():
    return get_disk_weight_storage()


def get_inmemory_weight_storage():
    logger = logging.getLogger("MemoryWStore")

    base_weights = None
    weights = {}
    rounds = set()

    def store_weights(metadata, weights):
        job_id = metadata.get("job_id", 0)
        weight_type = metadata.get("wt", "lw")
        round_id = metadata.get("round_id", 0)
        rounds.add(round_id)

        key = f"{job_id}-{round_id}-{weight_type}"

        weight_store = weights.get(key, [])
        weight_store.add(weights)
        weights[key] = weight_store

    def get_latest_weight_round(metadata):
        job_id = metadata.get("job_id", 0)
        round_id = max(rounds)
        key = f"{job_id}-{round_id}-lw"

        return weights.get(key, [])

    def ensure_default_weights(_, get_weights):
        logger.debug("Checking for base weights")

        # job_id = metadata.get("job_id", 0)
        if base_weights is None:
            base_weights = get_weights()

    sn = SimpleNamespace()
    sn.get_base_weights = lambda: base_weights
    sn.store_weights = store_weights
    sn.get_latest_weight_round = get_latest_weight_round
    sn.ensure_default_weights = ensure_default_weights

    return sn


def get_disk_weight_storage(base_path="./weights"):
    logger = logging.getLogger("DiskWStore")

    Path(base_path).mkdir(parents=True, exist_ok=True)

    def get_last_round():
        context = get_context()
        job_id = context.get("job_id", 0)
        return get_last_global_round_number(job_id, base_path)

    def store_weights(metadata, weights):
        job_id = metadata.get("job_id", 0)
        weight_type = metadata.get("wt", "lw")
        round_id = metadata.get("round_id", -1)
        partition_id = metadata.get("partition_id", 0)

        logger.info(
            "Storing weights %s %s %s %s", job_id, weight_type, round_id, partition_id
        )

        if round_id < 1:
            last_round = int(get_last_global_round_number(job_id, base_path))
            round_id = last_round + 1

        if weight_type == "lw":
            sender = metadata.get("sender", partition_id)
            samples = metadata.get("samples", 1)

            file_name = f"{job_id}_{round_id}_{samples}_{sender}_{weight_type}.bz2"
        else:
            file_name = f"{job_id}_{round_id}_{weight_type}.bz2"

        print("Saving weights to file", file_name)

        path = os.path.join(base_path, file_name)

        with bz2.BZ2File(path, "w") as file:
            np.savez(file, *weights)

    def base_weights_exist(metadata):
        job_id = metadata.get("job_id", 0)

        dw_file_name = f"{job_id}_gwdf.bz2"
        dw_file_path = os.path.join(base_path, dw_file_name)

        exists = os.path.exists(dw_file_path)

        if exists:
            logger.debug("Base weights file %s exists", dw_file_name)
        else:
            logger.debug("Base weights file %s does not exist", dw_file_name)

        return exists

    def create_base_weights(metadata, get_weights):
        job_id = metadata.get("job_id", 0)

        dw_file_name = f"{job_id}_gwdf.bz2"
        dw_file_path = os.path.join(base_path, dw_file_name)

        if os.path.exists(dw_file_path):
            logger.debug("Default weight file exists")
            return

        logger.debug("No default weights exist creating")
        weights = get_weights()

        with bz2.BZ2File(dw_file_path, "w") as file:
            np.savez(file, *weights)

    def ensure_default_weights(metadata, get_weights):
        logger.debug("Checking for base weights")

        if not base_weights_exist(metadata):
            create_base_weights(metadata, get_weights)

    def get_base_weights(metadata):
        job_id = metadata.get("job_id", 0)
        file_name = get_last_globalweight_file(job_id, base_path)

        if file_name is not None:
            logger.debug("Found base weights file %s", file_name)
            print("Found base weights file", file_name)
            file_path = os.path.join(base_path, file_name)
            return load_weights_from_file(file_path)

        dw_file_name = f"{job_id}_gwdf.bz2"
        dw_file_path = os.path.join(base_path, dw_file_name)

        if os.path.exists(dw_file_path):
            logger.debug("Loading weight from default weight file")
            print("Loading weights from file", dw_file_path)
            return load_weights_from_file(dw_file_path)

        logger.warning("Storage unable to find a base weight file for jobid %s", job_id)
        return None

    def get_latest_weight_round(metadata):
        job_id = metadata.get("job_id", 0)
        w_files = get_last_localweight_files(job_id, base_path)
        if w_files is None:
            return []
        logger.debug(
            "Found %s local weight files in weight round %s", len(w_files), w_files[0]
        )
        return [
            load_weights_from_file(os.path.join(base_path, file)) for file in w_files
        ]

    sn = SimpleNamespace()
    sn.get_base_weights = get_base_weights
    sn.store_weights = store_weights
    sn.get_latest_weight_round = get_latest_weight_round
    sn.ensure_default_weights = ensure_default_weights
    sn.create_base_weights = create_base_weights
    sn.base_weights_exist = base_weights_exist
    sn.get_last_round_number = get_last_round

    return sn


def get_last_round_number(job_id, partition, base_path="./weights"):
    last_files = get_last_partition_localweight_file(job_id, partition, base_path)

    if last_files is None:
        return -1

    return get_round_from_name(last_files[0])


def get_last_global_round_number(job_id, base_path="./weights"):
    last_file = get_last_globalweight_file(job_id, base_path)

    if last_file is None:
        return -1

    return get_round_from_name(last_file)


def get_disk_storage():
    sn = SimpleNamespace()

    def store_pickled(path, data):
        with open(path, "wb") as file:
            pickle.dump(data, file)

    def store_pickled_compress(path, data):
        with bz2.BZ2File(path, "w") as file:
            cPickle.dump(data, file)

    def load_pickled(path):
        with open(path, "rb") as file:
            data = pickle.load(file)
            return data

    def load_pickled_compress(path):
        with bz2.BZ2File(path, "rb") as file:
            data = cPickle.load(file)
            return data

    def load_data(path, compressed=True):
        if compressed:
            return load_pickled_compress(path)

        return load_pickled(path)

    def store_data(path, data, compressed=True):
        if compressed:
            store_pickled_compress(path, data)
        else:
            store_pickled(path, data)

    def store_json(path, data):
        with open(path, "w") as file:
            json.dump(data, file)

    sn.store_data = store_data
    sn.load_data = load_data
    sn.store_json = store_json

    return sn
