from copy import deepcopy

from .benchmark import BENCHMARK_BASE_CONFIG

BINARY_SEARCH_CONFIG = deepcopy(BENCHMARK_BASE_CONFIG)
BINARY_SEARCH_CONFIG.update(
    NAME="BinarySearch",
    MIN=32,
    MAX=32800,
)
