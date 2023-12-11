from copy import deepcopy
from easydict import EasyDict

from aqa.utils import Registry

BENCHMARKS = Registry("models")


def build_benchmark(config):
    config = deepcopy(config.BENCHMARK)
    benchmark_cls = BENCHMARKS[config.pop("NAME")]
    dataset_file = config.pop("DATASET_FILE")

    config = EasyDict({k.lower(): v for k, v in config.items()})
    benchmark = benchmark_cls(**config)
    benchmark.load_testcases_from_file(dataset_file)

    return benchmark
