import argparse
import glob
import json
import logging
import os

import matplotlib.pyplot as plt

import pandas as pd
import streamlit as st


def get_details_from_name(fname: str):
    parts = fname.split("_")
    partition_parts = parts[2:-1]
    if len(partition_parts) > 2:
        partition = "_".join(partition_parts[-2:])
    else:
        partition = "_".join(partition_parts)
    return parts[1], partition, parts[0]


def partition_names(fname: str):
    parts = fname.split("_")
    partition_parts = parts[2:-1]
    if len(partition_parts) > 2:
        partition = "_".join(partition_parts[-2:])
    else:
        partition = "_".join(partition_parts)
    return "_".join([parts[0], partition])


def get_partition_from_name(fname: str):
    parts = fname.split("_")
    return "_".join(parts[2:-1])


def get_round_from_name(fname: str):
    parts = fname.split("_")
    return parts[1]


def get_report_files(base_path, job_no=None):
    filter_string = "*report.json" if job_no is None else f"{job_no}_*report.json"
    search = os.path.join(base_path, filter_string)
    files = glob.glob(search)
    file_names = [os.path.basename(file) for file in files]
    ordered = sorted(file_names, key=get_round_from_name, reverse=False)
    return ordered


def sort_tuple(name):
    parts = name.split("_")
    parts = list([int(p) if p.isdigit() else p for p in parts])
    return (parts[0], parts[0] if len(parts) < 2 else parts[1], name)


def sorted_names(name_list):
    return list(map(lambda x: x[-1], sorted(map(sort_tuple, name_list))))


def line_chart(ds: pd.DataFrame, round_column: str = "loss"):
    margins = 0.05

    # fig = plt.figure(figsize=(6, 4))  # a new figure window, figsize goes here
    fig = plt.figure()  # a new figure window, figsize goes here
    ax = fig.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)

    dsg = ds.groupby("partition")
    legends = []
    for name, group in dsg:
        legends.append(name)
        ax.plot(group["epoch"], group[round_column])

    # decorate then sort
    legends = sorted_names(legends)

    if len(legends) > 4:
        ax.legend(legends[0 : min(12, len(legends))], fancybox=True, framealpha=0.5)
    else:
        ax.legend(legends)
    ax.set_xlabel("Training epoch")
    ax.set_ylabel(round_column)
    ax.set_title(f"Training {round_column}")

    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    plt.gca().set_xticks(list(range(ds["epoch"].min(), ds["epoch"].max() + 1)))

    # if tight:
    #     plt.tight_layout(h_pad=3)

    plt.margins(margins)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Display results from storage learning jobs"
    )
    parser.add_argument(
        "-jid",
        action="store",
        dest="job_id",
        default=-1,
        type=int,
        help="Job id of the current learning job",
    )
    parser.add_argument(
        "-report-dir",
        action="store",
        dest="report_dir",
        default="reports",
        type=str,
        help="Job id of the current learning job",
    )
    arg_list = parser.parse_args()

    logging.info("Running result visualisation for %s", arg_list.job_id)

    st.title("Storage learner results")

    all_files = get_report_files(arg_list.report_dir)

    st.sidebar.title = "Parameters"
    job_ids = sorted(list({int(f.split("_")[0]) for f in all_files}))

    if arg_list.job_id >= 0:
        initial_index = job_ids.index(arg_list.job_id)
        jobs = st.sidebar.multiselect("Choose job to display", job_ids, initial_index)
    else:
        jobs = st.sidebar.multiselect("Choose job to display", job_ids)
    job_strings = [f"{j}_" for j in jobs]

    files = [f for f in all_files if any(map(f.startswith, job_strings))]

    partitions = sorted_names({partition_names(f) for f in files})
    partition_select = st.sidebar.multiselect("Partitions", partitions)

    files = [f for f in files if partition_names(f) in partition_select]

    data = {
        "epoch": [],
        "partition": [],
    }

    keys = None

    epochs = {}
    for file in files:
        with open(os.path.join(arg_list.report_dir, file)) as f:
            from_file = json.load(f)
            round_id, partition, job = get_details_from_name(file)
            partition_name = f"{job}_{partition}"
            keys = list(from_file.keys())
            for key in keys:
                if key not in data:
                    data[key] = []

            for i in range(0, len(from_file[keys[0]])):
                epoch = epochs.get(partition_name, 1)
                data["partition"].append(partition_name)
                data["epoch"].append(int(epoch))
                for key in keys:
                    data[key].append(from_file[key][i])
                epoch += 1
                epochs[partition_name] = epoch

    result_data = pd.DataFrame(data=data)

    for key in keys or []:
        line_chart(result_data, round_column=key)
        st.pyplot()

    st.write(result_data.sort_values(by=["partition", "epoch"]))
