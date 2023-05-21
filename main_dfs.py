import models
from benchmarks import DFSEvaluator
import matplotlib.pyplot as plt
import numpy as np

NUM = 5
MODEL = "gpt-3.5-turbo"
# MODEL = "text-davinci-002"
# TF = "l0"
MCQ = False
EXPLAIN_ALGO = True
STATED = False

evaluator = DFSEvaluator(
    node_num=NUM, explain_algo=EXPLAIN_ALGO, mcq=MCQ, provide_state=STATED,
    format_tolerant=True, max_retry=3, max_step=15
)
model = models.ChatGPT(MODEL)
# model = models.Davinci("text-davinci-003")
# model = models.DfsVerifier()
# model = models.DFSModel()

# model = models.Llama(
#     "/data3/siwei/my_llama/tokenizer/",
#     "/data3/siwei/my_llama/vicuna/vicuna-7b"
# )

# metric, full_result = evaluator.test_one_time(
#     model,
#     teacher_forcing=False,
# )

metric, full_result = evaluator.test_multi_time(model, 5, teacher_forcing_mode="l2")

# print(metric)

# trial_name = f"./dfs_{MODEL.replace('.', '-')}_{TF}"
# if MCQ:
#     trial_name += "_mcq"
# if EXPLAIN_ALGO:
#     trial_name += "_explained"
# if STATED:
#     trial_name += "_stated"
