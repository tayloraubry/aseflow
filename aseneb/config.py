from pathlib import Path
from dataclasses import dataclass
from typing import Optional

@dataclass
class PathsConfig:
    workdir: Path

@dataclass
class NEBConfig:
    climb: bool = True
    method: Optional[str] = None
    interpolation_method: Optional[str] = None
    mic: bool = True

@dataclass
class OptimizerConfig:
    name: str
    fmax: Optional[float] = None
    steps: Optional[int] = None
    a: Optional[float] = None

@dataclass
class CalculatorConfig:
    type: str
    model: Optional[str] = None
    device: Optional[str] = None

@dataclass
class RunConfig:
    paths: PathsConfig
    neb: NEBConfig
    optimizer: Optional[OptimizerConfig] = None
    calculator: Optional[CalculatorConfig] = None