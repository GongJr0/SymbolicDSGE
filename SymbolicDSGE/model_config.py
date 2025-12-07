import yaml
from dataclasses import dataclass, asdict
from typing import Any


class FlowList(list):
    pass


def makeflow(
    dumper: yaml.representer.BaseRepresenter, obj: FlowList
) -> yaml.nodes.SequenceNode:
    return dumper.represent_sequence("tag:yaml.org,2002:seq", obj, flow_style=True)


yaml.add_representer(FlowList, makeflow)


@dataclass
class Symbols:
    variables: list[str]
    shocks: list[str]
    parameters: list[str]


@dataclass(init=False)
class SnowDropConfig:
    NAME: str
    SYMBOLS: Symbols
    EQUATIONS: list[str]
    CALIBRATION: dict[str, float]
    OPTIONS: dict[str, Any]

    def __init__(self, filepath: str) -> None:
        with open(filepath, "r") as file:
            data = yaml.safe_load(file)

        self.NAME = data["name"]
        self.SYMBOLS = Symbols(**data["symbols"])
        self.EQUATIONS = data["equations"]
        self.CALIBRATION = data["calibration"]
        opt: dict[str, Any] = data.get("options", {})
        for k, v in opt.items():
            if isinstance(v, list):
                opt[k] = FlowList(v)
        self.OPTIONS = opt

        self.validate_calib()
        return

    def validate_calib(self) -> bool:
        all_params = {
            *self.SYMBOLS.parameters,
            *self.SYMBOLS.shocks,
            *self.SYMBOLS.variables,
        }
        missing_params = [
            param for param in all_params if param not in self.CALIBRATION
        ]

        msg = ""
        if missing_params:
            msg += f"Missing parameters in calibration: {missing_params}.\n "

        if msg:
            raise ValueError(msg)
        return True

    def to_yaml(self, filepath: str) -> None:
        out = {k.lower(): v for k, v in asdict(self).items()}
        with open(filepath, "w") as file:
            yaml.dump(out, file)
        return
