import yaml
from pathlib import Path
from aseneb.config import *


def load_config(path):
    with open(path) as f:
        data = yaml.safe_load(f)

    # Use current working directory if "paths" is missing
    workdir = Path(data.get("paths", {}).get("workdir", Path.cwd()))

    return RunConfig(
        paths=PathsConfig(workdir=workdir),
        neb=NEBConfig(
            #n_images=data["neb"]["n_images"],
            climb=data["neb"]["climb"],
            method=data["neb"]["method"],
            #initial=Path(data["neb"]["initial"]),
            #final=Path(data["neb"]["final"]),
            interpolation_method=data["neb"].get("interpolation_method", "idpp"),
            mic=data["neb"].get("mic", True),
        ),
        optimizer=OptimizerConfig(**data["optimizer"]),
        calculator=CalculatorConfig(**data["calculator"]),
    )