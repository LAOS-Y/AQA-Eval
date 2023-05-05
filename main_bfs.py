import models
from benchmarks import BfsEvaluator

NUM = 30
MODEL = "gpt-3.5-turbo"
# MODEL = "text-davinci-002"
TF = "l0"
MCQ = False
EXPLAIN_ALGO = True
STATED = True

evaluator = BfsEvaluator()
model = models.ChatGPT()
# model = models.Davinci("text-davinci-003")
# model = models.BFSModel()
evaluator.test_one_time(model, teacher_forcing=True, mcq=MCQ, explain_algo=EXPLAIN_ALGO, provide_state=STATED)