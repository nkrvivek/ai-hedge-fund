"""Alpha models — view-forming components of the quant stack.

See v2/signals/base.py for the AlphaModel / QuantModel interface.
Concrete models register here as they are implemented.
"""

from __future__ import annotations

from v2.signals.base import AlphaModel, QuantModel
from v2.signals.buffett import BuffettAgent
from v2.signals.burry import BurryAgent
from v2.signals.damodaran import DamodaranAgent
from v2.signals.graham import GrahamAgent
from v2.signals.lynch import LynchAgent
from v2.signals.munger import MungerAgent
from v2.signals.wood import WoodAgent
from v2.signals.llm_agent import LLMAgent
from v2.signals.pead import PEADModel

ALPHA_MODEL_REGISTRY: dict[str, type[AlphaModel]] = {
    "pead": PEADModel,
    "buffett": BuffettAgent,
    "damodaran": DamodaranAgent,
    "munger": MungerAgent,
    "burry": BurryAgent,
    "wood": WoodAgent,
    "lynch": LynchAgent,
    "graham": GrahamAgent,
}

__all__ = [
    "AlphaModel",
    "QuantModel",
    "LLMAgent",
    "BuffettAgent",
    "PEADModel",
    "ALPHA_MODEL_REGISTRY",
]
