from copy import deepcopy
from easydict import EasyDict

from aqa.utils import Registry

BENCHMARKS = Registry("models")


def build_benchmark(config):
    config = deepcopy(config.BENCHMARK)
    model_cls = BENCHMARKS[config.NAME]

    config.pop("NAME")
    config = EasyDict({k.lower(): v for k, v in config.items()})
    model = model_cls(**config)

    return model
