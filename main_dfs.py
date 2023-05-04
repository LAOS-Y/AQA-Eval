import models
from benchmarks import DFSEvaluator
import matplotlib.pyplot as plt
import numpy as np

NUM = 30
MODEL = "gpt-3.5-turbo"
# MODEL = "text-davinci-002"
TF = "l0"
MCQ = True
EXPLAIN_ALGO = True
STATED = True

evaluator = DFSEvaluator()
# model = models.ChatGPT(MODEL)
# model = models.Davinci("text-davinci-003")
# model = models.DfsVerifier()
model = models.DFSModel()
# evaluator.test_multi_time(model, NUM, teacher_forcing_mode=TF, mcq=MCQ, explain_algo=EXPLAIN_ALGO, provide_state=STATED)
evaluator.test_one_time(
    model,
    teacher_forcing=True,
    explain_algo=EXPLAIN_ALGO,
    mcq=MCQ,
    provide_state=STATED
)

trial_name = f"./dfs_{MODEL.replace('.', '-')}_{TF}"
if MCQ:
    trial_name += "_mcq"
if EXPLAIN_ALGO:
    trial_name += "_explained"
if STATED:
    trial_name += "_stated"
