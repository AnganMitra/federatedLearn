import argparse
import json
import os

from context import set_context, get_context
from configuration import DefaultPorts

BIND_TEMPLATE = "tcp://*:%s"
CONNECT_TEMPLATE = "tcp://127.0.0.1:%s"

default_pid = int(os.environ.get("INSTANCE_ID", 1)) - 1
default_wsp = os.environ.get("WEIGHT_STORAGE_PROVIDER", "disk_storage")
default_rsp = os.environ.get("REPORT_STORAGE_PROVIDER", "disk_storage")

default_reporter_address = os.environ.get(
    "REPORTER_ADDRESS", CONNECT_TEMPLATE % DefaultPorts.reporter
)
default_ll_bind_address = os.environ.get(
    "LL_BIND_ADDRESS", BIND_TEMPLATE % DefaultPorts.local_learner
)

default_control_address = os.environ.get(
    "LL_CONTROL_ADDRESS", BIND_TEMPLATE % DefaultPorts.controller
)

default_gl_bind_address = os.environ.get(
    "GL_BIND_ADDRESS", BIND_TEMPLATE % DefaultPorts.global_learner
)

default_control_bind_address = os.environ.get(
    "CL_BIND_ADDRESS", BIND_TEMPLATE % DefaultPorts.controller
)

default_global_address = os.environ.get(
    "GLOBAL_ADDRESS", CONNECT_TEMPLATE % DefaultPorts.global_learner
)

default_pub_address = os.environ.get("PUB_ADDRESS", CONNECT_TEMPLATE % DefaultPorts.pub)

default_sub_address = os.environ.get("SUB_ADDRESS", CONNECT_TEMPLATE % DefaultPorts.sub)

default_pub_bind = os.environ.get("PUB_BIND", BIND_TEMPLATE % DefaultPorts.sub)

default_sub_bind = os.environ.get("SUB_BIND", BIND_TEMPLATE % DefaultPorts.pub)


def create_parser(description):
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument(
        "-c",
        "--config-file",
        action="store",
        dest="config_file",
        default="config.json",
        help="Name of configuration file",
    )

    return parser


def get_args(parser):
    args = parser.parse_args()
    if os.path.exists(args.config_file):
        with open(args.config_file, "rt") as f:
            t_args = argparse.Namespace()
            t_args.__dict__.update(json.load(f))
            args = parser.parse_args(namespace=t_args)
    return args


def bind_args_to_context(arg_list):
    old = get_context()
    context = {k: getattr(arg_list, k) for k in vars(arg_list)}
    set_context({**old, **context})


def add_remote_log(parser):
    parser.add_argument(
        "-rl", action="store_true", dest="remote_log", help="Enable remote logging",
    )
    parser.add_argument(
        "-rlp",
        action="store",
        dest="remote_log_proxy",
        default=True,
        help="Enable remote logging proxy",
    )
    parser.add_argument(
        "-rlt",
        action="store",
        dest="remote_log_topic",
        default="FedableLogger",
        help="Remote log topic",
    )
    parser.add_argument(
        "-rlc",
        action="store",
        dest="remote_log_connect",
        default="tcp://127.0.0.1:8880",
        help="Remote log connect address",
    )
    parser.add_argument(
        "-rllla",
        action="store",
        dest="remote_log_bind_back",
        default="tcp://*:8881",
        help="Remote log proxy log listener address",
    )
    parser.add_argument(
        "-rllca",
        action="store",
        dest="remote_log_bind_front",
        default="tcp://*:8880",
        help="Remote log proxy logger connection address",
    )


def add_pid(parser):
    parser.add_argument(
        "-pid",
        metavar="Partition Id",
        action="store",
        dest="partition_id",
        default=default_pid,
        type=int,
        help="Partition number for the data set",
    )


def add_iterations(parser):
    parser.add_argument(
        "-i",
        metavar="Iterations",
        action="store",
        dest="iterations",
        default=1,
        type=int,
        help="Number of learning rounds",
    )


def add_inproc(parser):
    parser.add_argument(
        "-inproc",
        action="store_true",
        dest="inproc",
        help="Run global server in proc",
    )


def add_partitions(parser):
    parser.add_argument(
        "-partitions",
        metavar="No Partitions",
        action="store",
        dest="partitions",
        default=0,
        type=int,
        help="Number of partitions to use",
    )


def add_learn_partitions(parser):
    parser.add_argument(
        "-parts",
        metavar="Partitions to learn",
        action="store",
        nargs="+",
        dest="parts",
        default=[],
        type=int,
        help="Ids of partitions to use",
    )


def add_jid(parser):
    parser.add_argument(
        "-jid",
        metavar="Job Id",
        action="store",
        dest="job_id",
        required=True,
        type=int,
        help="Job id of the current learning job",
    )


def add_rid(parser):
    parser.add_argument(
        "-rid",
        metavar="Round Id",
        action="store",
        dest="round_id",
        default=-1,
        type=int,
        help="Round id of the current learning job",
    )


def add_ignorekill(parser):
    parser.add_argument(
        "-ik",
        metavar="Ignore kill",
        action="store",
        dest="ignore_kill",
        default=False,
        help="Ignore kill messages",
    )


def add_num_learners(parser):
    parser.add_argument(
        "-learners",
        metavar="Num learners",
        action="store",
        dest="num_learners",
        default=1,
        help="Number of learners",
    )


def add_instance_name(parser):
    parser.add_argument(
        "-name",
        action="store",
        dest="round_id",
        default=None,
        type=int,
        help="Name of the running instance",
    )


def add_verbose(parser):
    parser.add_argument(
        "-v",
        metavar="Verbose learning",
        action="store",
        dest="verbose",
        required=False,
        default=0,
        type=int,
        help="Verbose training mode",
    )


def add_epochs(parser):
    parser.add_argument(
        "-epochs",
        action="store",
        dest="epochs",
        default=1,
        type=int,
        help="Number of epochs to run for learner",
    )


def add_report_dir(parser):
    parser.add_argument(
        "-report-dir",
        action="store",
        dest="report_dir",
        default="reports",
        type=str,
        help="Directory to store reports in report storage provider",
    )


def add_weight_dir(parser):
    parser.add_argument(
        "-weight-dir",
        action="store",
        dest="weight_dir",
        default="weights",
        type=str,
        help="Directory to store weights in weight storage provider",
    )


def add_pubsub_address(parser):
    add_pub_address(parser)
    add_sub_address(parser)


def add_pubsub_bind(parser):
    add_pub_bind(parser)
    add_sub_bind(parser)


def add_pub_address(parser):
    parser.add_argument(
        "-pub-address",
        action="store",
        dest="pub_address",
        default=default_pub_address,
        type=str,
        help="Address to connect message bus publisher",
    )


def add_sub_address(parser):
    parser.add_argument(
        "-sub-address",
        action="store",
        dest="sub_address",
        default=default_sub_address,
        type=str,
        help="Address to connect message bus subscriber",
    )


def add_pub_bind(parser):
    parser.add_argument(
        "-pub-bind",
        action="store",
        dest="pub_bind",
        default=default_pub_bind,
        type=str,
        help="Address to bind message bus publisher",
    )


def add_sub_bind(parser):
    parser.add_argument(
        "-sub-bind",
        action="store",
        dest="sub_bind",
        default=default_sub_bind,
        type=str,
        help="Address to bind message bus subscriber",
    )


def add_storage_dirs(parser):
    add_report_dir(parser)
    add_weight_dir(parser)


def add_storage_provider(parser):
    parser.add_argument(
        "-wsp",
        action="store",
        dest="storage_provider",
        default=default_wsp,
        type=str,
        help="Storage provider for weights",
    )


def add_data_provider(parser):
    parser.add_argument(
        "-dp",
        action="store",
        dest="data_provider",
        required=True,
        type=str,
        help="Data provider for the learner",
    )


def add_fedavg_provider(parser):
    parser.add_argument(
        "-fap",
        action="store",
        dest="fedavg_algo",
        default="always_fedavg",
        type=str,
        help="Federated averaging algorithm",
    )


def add_model_provider(parser):
    parser.add_argument(
        "-mp",
        action="store",
        dest="model_provider",
        required=True,
        type=str,
        help="Model provider for the learner",
    )


def add_report_storage_provider(parser):
    parser.add_argument(
        "-rsp",
        action="store",
        dest="report_storage_provider",
        default=default_rsp,
        type=str,
        help="Storage provider for reports",
    )


def add_reporter_address(parser):
    parser.add_argument(
        "-reporter-address",
        action="store",
        dest="reporter_address",
        default=default_reporter_address,
        type=str,
        help="Address to connect to reporter",
    )


def add_ll_bind_address(parser):
    parser.add_argument(
        "-bind-address",
        action="store",
        dest="bind_address",
        default=default_ll_bind_address,
        type=str,
        help="Bind address for local learner",
    )


def add_gl_bind_address(parser):
    parser.add_argument(
        "-bind-address",
        action="store",
        dest="bind_address",
        default=default_gl_bind_address,
        type=str,
        help="Bind address for global learner",
    )


def add_control_connect_address(parser):
    parser.add_argument(
        "-control-address",
        action="store",
        dest="control_address",
        default=default_control_address,
        type=str,
        help="Address to connect to control local learner",
    )


def add_control_bind_address(parser):
    parser.add_argument(
        "-control-address",
        action="store",
        dest="control_address",
        default=default_control_bind_address,
        type=str,
        help="Address to bind controller",
    )


def add_global_learner_address(parser):
    parser.add_argument(
        "-global-address",
        action="store",
        dest="global_address",
        default=default_global_address,
        type=str,
        help="Address to connect to global learner",
    )


def add_local_learner_addresses(parser):
    add_reporter_address(parser)
    add_ll_bind_address(parser)
    add_control_connect_address(parser)
    add_global_learner_address(parser)


def add_global_learner_addresses(parser):
    add_gl_bind_address(parser)
    add_control_connect_address(parser)


def create_byroundstorage_parser(description):
    parser = create_parser(description)
    add_remote_log(parser)
    add_pid(parser)
    add_jid(parser)
    add_rid(parser)
    add_epochs(parser)
    add_storage_dirs(parser)
    return parser


def byroundstorage_args(description):
    parser = create_byroundstorage_parser(description)
    return parser.parse_args()
