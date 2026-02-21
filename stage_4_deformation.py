from dataclasses import dataclass, field
from enum import Enum
import numpy as np


class ZStart(Enum):
    X = "geo_lineX"
    Y = "geo_lineY"

@dataclass
class ZSequence:
    StartLineX: bool
    TipPt: int
    PolyDatArr: list
    TopCorner: bool
    BotCorner: bool
