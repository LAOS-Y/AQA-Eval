import json
import rich
from loguru import logger

import models
from benchmarks import BinarySearchEvaluator

evaluator = BinarySearchEvaluator(0, 2**10, format_tolerant=True, max_retry=3, max_guess=15)
model = models.Llama(
    "/data3/siwei/my_llama/tokenizer/",
    "/data3/siwei/my_llama/vicuna/vicuna-7b"
)
logger.info("Start testing Llama model with teacher forcing.")
err, full_result = evaluator.test_multi_time(model, 2, teacher_forcing_mode="l2")
rich.print("Full result:", full_result)
rich.print("Err of Llama:", err)

json.dump(full_result, open("vicuna-7b-l2.json", mode="w"))
