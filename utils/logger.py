import os.path as osp
from loguru import logger

from .file import ensure_dir


def setup_logger(filename=None):
    ensure_dir(osp.dirname(filename))
    logger.add(filename)
