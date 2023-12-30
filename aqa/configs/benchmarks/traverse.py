from copy import deepcopy

from .benchmark import BENCHMARK_BASE_CONFIG

BFS_CONFIG = deepcopy(BENCHMARK_BASE_CONFIG)
BFS_CONFIG.update(
    NAME="BFS",
    NODE_NUM=15,
    EXPLAIN_ALGO=True,
    MCQ=False,
    PROVIDE_STATE=False,
)

DFS_CONFIG = deepcopy(BENCHMARK_BASE_CONFIG)
DFS_CONFIG.update(
    NAME="DFS",
    NODE_NUM=8,
    EXPLAIN_ALGO=True,
    MCQ=False,
    PROVIDE_STATE=False,
)
