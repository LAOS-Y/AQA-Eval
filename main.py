import argparse
from loguru import logger

from aqa.utils import dynamic_import, eval

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", type=str, required=True)

    args = parser.parse_args()

    config = dynamic_import(args.config).config
    logger.info(config)
    eval(config)
