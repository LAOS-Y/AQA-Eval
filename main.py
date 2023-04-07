from loguru import logger

import models
from benchmarks import BinarySearchEvaluator

evaluator = BinarySearchEvaluator(5, 20)

model = models.SimpleModel(5, 20)
logger.info("Start testing SimpleModel.")
count = evaluator.test_one_time(model)
logger.info("Total guess count of SimpleModel: {}".format(count))

model = models.BSModel(5, 20)
logger.info("Start testing BSModel.")
count = evaluator.test_one_time(model)
logger.info("Total guess count of BSModel: {}".format(count))

evaluator = BinarySearchEvaluator(5, 2**10)
model = models.BLOOMZ(name="bigscience/bloomz-7b1")
logger.info("Start testing BLOOMZ model with teacher forcing.")
err = evaluator.tf_test_one_time(model)
logger.info("Err of BLOOMZ: {}".format(err))
