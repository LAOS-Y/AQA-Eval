import models
from benchmarks import BinarySearchEvaluator

evaluator = BinarySearchEvaluator(5, 20)

print("Start testing SimpleModel.")
model = models.SimpleModel(5, 20)
count = evaluator.test_one_time(model, verbose=True)
print("Total guess count of SimpleModel: {}".format(count))

print("Start testing BSModel.")
model = models.BSModel(5, 20)
count = evaluator.test_one_time(model, verbose=True)
print("Total guess count of BSModel: {}".format(count))
