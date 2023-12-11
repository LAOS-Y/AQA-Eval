from .benchmarks import BENCHMARK_BASE_CONFIG
from .config import Config
from .models import FASTCHAT_MODEL_CONFIG

BASE_CONFIG = Config(
    BENCHMARK=BENCHMARK_BASE_CONFIG,
    EVAL=Config(
        TIMES=400,
        NUM_EXAMPLES=0,
        TEACHER_FORCING=False,
        RESUME=False,
    ),
    MODEL=FASTCHAT_MODEL_CONFIG,
)
