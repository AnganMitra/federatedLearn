from dataclasses import dataclass

# defines which ports to listen to framework components like reporter, local learner, global learner, controller, publisher and subscriber.

@dataclass(frozen=True)
class DefaultPorts:
    reporter: int = 5050
    local_learner: int = 5060
    global_learner: int = 5070
    controller: int = 5080
    pub: int = 5052
    sub: int = 5051
