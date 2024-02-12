from copy import deepcopy

from .benchmark import BENCHMARK_BASE_CONFIG

CAVE_BFS_CONFIG = deepcopy(BENCHMARK_BASE_CONFIG)
CAVE_BFS_CONFIG.update(
    NAME="CaveBFS",
    NODE_NUM=15,
    EXPLAIN_ALGO=False,
    MCQ=False,
    PROVIDE_STATE=False,
)

CAVE_DFS_CONFIG = deepcopy(BENCHMARK_BASE_CONFIG)
CAVE_DFS_CONFIG.update(
    NAME="CaveDFS",
    NODE_NUM=8,
    EXPLAIN_ALGO=False,
    MCQ=False,
    PROVIDE_STATE=False,
)
