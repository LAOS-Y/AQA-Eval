import models
import rich
from benchmarks import BFSEvaluator

NUM = 30
MODEL = "gpt-3.5-turbo"
# MODEL = "text-davinci-002"
TF = "l0"
MCQ = False
EXPLAIN_ALGO = True
STATED = True

evaluator = BFSEvaluator(node_num=5, use_scene_instruction=True)
model = models.ChatGPT(MODEL)
# model = models.Davinci("text-davinci-003")
# model = models.BFSModel()
# evaluator.test_one_time(model, teacher_forcing=True, mcq=MCQ, explain_algo=EXPLAIN_ALGO, provide_state=STATED)
err, full_result = evaluator.test_multi_time(model, 1)
rich.print("Full result:", full_result)
rich.print("Err of Llama:", err)