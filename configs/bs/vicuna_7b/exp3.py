from copy import deepcopy

from aqa.configs import benchmarks, models
from aqa.configs.base_config import BASE_CONFIG

config = deepcopy(BASE_CONFIG)
config.update(
    BENCHMARK=dict(
        benchmarks.BINARY_SEARCH_CONFIG,
        DATASET_FILE="binary_search_0.json",
        OUTPUT_DIR=__file__.replace("configs", "results", 1)
    ),
    MODEL=models.VICUNA_V15_7B_16K_CONFIG,
    EVAL=dict(
        NUM_EXAMPLES=3,
    )
)
