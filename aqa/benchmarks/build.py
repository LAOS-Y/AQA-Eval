from copy import deepcopy

import os
import os.path as osp
import aqa
from aqa.utils import Registry

BENCHMARKS = Registry("models")


def build_benchmark(config):
    config = deepcopy(config.BENCHMARK)
    benchmark_cls = BENCHMARKS[config.pop("NAME")]
    dataset_file = config.pop("DATASET_FILE")
    dataset_root = os.environ.get(
        "AQA_DATASET_ROOT",
        default="/".join(aqa.__file__.split("/")[:-2] + ["datasets"])
    )
    dataset_file = osp.join(dataset_root, dataset_file)

    config = dict({k.lower(): v for k, v in config.items()})
    benchmark = benchmark_cls(**config)
    benchmark.load_testcases_from_file(dataset_file)

    return benchmark
