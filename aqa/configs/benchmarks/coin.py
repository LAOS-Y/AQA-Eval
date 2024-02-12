from copy import deepcopy

from .benchmark import BENCHMARK_BASE_CONFIG

COIN_CONFIG = deepcopy(BENCHMARK_BASE_CONFIG)
COIN_CONFIG.update(
    NAME="Coin",
    MIN=32,
    MAX=32800,
)
