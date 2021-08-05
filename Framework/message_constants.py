from dataclasses import dataclass

"""
Identifying keywords by message, topic and bus status types.
"""
@dataclass
class MessageTypes:
    training_report: int = 0
    kill: str = "k"
    quit: str = "Q"
    worker_quit: str = "q"
    req_worker_quit: str = "rwq"
    all_kill: str = "ak"
    global_kill: str = "gk"
    reporter_kill: str = "rk"
    ping: str = "ping"
    pong: str = "pong"
    glping: str = "glping"
    glping2: str = "glping2"
    send_weights: str = "sw"
    reset_weights: str = "rw"
    req_fedavg: str = "rfa"
    learn: str = "l"
    multi_learn: str = "ml"
    learn_all: str = "la"


@dataclass
class Topics:
    learning_reports: str = "lreport"
    control: str = "control"
    global_worker_status: str = "gwstatus"
    local_learner_status: str = "llstatus"
    status_request: str = "statusrequest"
    local_learner: str = "ll"
    pid: str = "pid"


@dataclass
class BusStatus:
    failed: int = -1
    unknown: int = 0
    initializing: int = 1
    running: int = 2
    shutting_down: int = 3
    finished: int = 4
