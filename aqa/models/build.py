from copy import deepcopy
from easydict import EasyDict

from aqa.utils import Registry

MODELS = Registry("models")


def build_model(config):
    config = deepcopy(config.MODELS)
    model_cls = MODELS[config.NAME]

    config.pop("NAME")
    config = EasyDict({k.lower(): v for k, v in config.items()})
    model = model_cls(**config)

    return model
