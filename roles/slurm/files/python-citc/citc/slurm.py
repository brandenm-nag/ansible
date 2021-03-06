#! /usr/bin/env python

import pathlib
import subprocess
from typing import Iterator, Type, Optional


def node_list(slurm_conf: pathlib.Path) -> Iterator[str]:
    """
    Given a config file, gives all the nodes listed within
    """
    with slurm_conf.open() as conf:
        for line in conf:
            if line.startswith("NodeName="):
                nodelist = line.split()[0][9:]
                if "[" in nodelist:
                    # TODO use pyslurm.hostlist().create()/get_list()
                    pass
                yield from nodelist.split(",")


NODE_STATE_FLAGS = {
    "*": "not responding",
    "~": "power save",
    "#": "powering up",
    "%": "powering down",
    "$": "main reservation",
    "@": "pending reboot",
}


class SlurmNode:
    SINFO_FIELDS = {
        "nodelist": "%N",
        "statelong": "%T",
        "reason": "%E",
        "features": "%f",
    }

    name: str
    state: str
    state_flag: Optional[str]
    features: dict
    reason: str

    def __init__(self, name, state, features, state_flag, reason):
        self.name = name
        self.state = state
        self.state_flag = state_flag
        self.features = features
        self.reason = reason

    @classmethod
    def from_name(cls: Type["SlurmNode"], nodename: str) -> "SlurmNode":
        sinfo_format = "|".join(f"{k}:{v}" for (k,v) in cls.SINFO_FIELDS.items())
        out = subprocess.run(
            ["sinfo", "--nodes", nodename, "--format", sinfo_format, "--noheader"],
            timeout=5,
            stdout=subprocess.PIPE,
        ).stdout.decode()
        data = {k: v for (k,v) in [x.split(':', 1) for x in out.split('|')]}
        if data["statelong"][-1] in NODE_STATE_FLAGS:
            state = data["statelong"][:-1]
            state_flag: Optional[str] = data["statelong"][-1]
        else:
            state = data["statelong"]
            state_flag = None

        features = parse_features(data["features"])
        reason = data["reason"]

        return cls(
            name=nodename,
            state=state,
            state_flag=state_flag,
            features=features,
            reason=reason,
        )

    def resume(self):
        subprocess.run(
            ["scontrol", "update", f"NodeName={self.name}", "state=Resume"], timeout=5
        )


def parse_features(feature_string: str) -> dict:
    feature_dict = {}
    for pair in feature_string.split(","):
        k, v = pair.split("=")
        feature_dict[k] = v
    return feature_dict


def all_nodes(slurm_conf):
    return [SlurmNode.from_name(hostname) for hostname in node_list(slurm_conf)]
