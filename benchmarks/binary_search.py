import random
from loguru import logger

from utils import DialogLogger

from models import BSModel


class BinarySearchEvaluator():
    def __init__(self, min=0, max=100):
        self.min = min
        self.max = max
        self.dialog_logger = DialogLogger(order=["Q", "A", "T"])

        self.teacher = BSModel(min, max)

    @property
    def init_prompt(self):
        return "You are required to guess the random number which I have just picked between {} and {}.\n" \
               "I will only give responses such as 'The true number is bigger than this guess' or 'The true number is smaller than this guess' or 'The true number is equal to this guess'.\n" \
               "Adjust your guess according to my response.\n" \
               "Try as few times as you can.\n" \
               "Start guessing after receiving 'START' command.\n" \
               "Stop guessing after receiving 'STOP' command.\n" \
               "Reply 'OK' if you understand.".format(self.min, self.max)

    def init_model(self, model, teacher_forcing=False):
        self.dialog_logger.info(Q=self.init_prompt)

        reply = model(self.init_prompt)
        if teacher_forcing:
            self.dialog_logger.info(A=reply, T="OK")
            return True

        self.dialog_logger.info(A=reply)
        return reply.strip() == "OK"

    def get_prompt(self, guess, target):
        if guess < target:
            return "The true number is bigger than this guess"
        if guess > target:
            return "The true number is smaller than this guess"

        return "The true number is equal to this guess"

    def is_valid(self, guess):
        try:
            guess = int(guess)
            return self.min <= guess and guess <= self.max
        except ValueError:
            return False

    def get_teacher_outputs(self, target):
        self.teacher.reset()
        self.init_model(self.teacher)
        qa_list = [(self.init_prompt, "OK")]

        guess = None
        guess_list = []
        prompt = "START"

        while guess != target:
            guess = self.teacher(prompt)
            guess_list.append(guess)
            qa_list.append((prompt, guess))

            prompt = self.get_prompt(guess, target)

        qa_list.append((prompt, ""))

        return qa_list

    def test_one_time(self, model):
        if not self.init_model(model):
            raise ValueError("Invalid Reply")

        target = random.randint(self.min, self.max)
        logger.info("Picked Random Number: {}".format(target))

        guess = None
        guess_list = []
        prompt = "START"

        while guess != target:
            self.dialog_logger.info(Q=prompt)

            guess = model(prompt)
            guess_list.append(guess)

            if not self.is_valid(guess):
                raise ValueError(f"Invalid Reply: {guess}")

            self.dialog_logger.info(A=guess)

            prompt = self.get_prompt(guess, target)

        self.dialog_logger.info(Q=prompt)

        return len(guess_list)

    def tf_test_one_time(self, model):
        target = random.randint(self.min, self.max)
        logger.info("Picked Random Number: {}".format(target))

        guess = None
        guess_list = []
        teacher_guess_list = []

        teacher_qa_list = self.get_teacher_outputs(target)

        for prompt, teacher_guess in teacher_qa_list[:-1]:
            self.dialog_logger.info(Q=prompt)

            guess = model(prompt)
            model.teacher_force(teacher_guess)
            guess_list.append(guess)
            teacher_guess_list.append(teacher_guess)

            self.dialog_logger.info(A=guess, T=teacher_guess)

        self.dialog_logger.info(Q=teacher_qa_list[-1][0])

        return self.calc_err(guess_list, teacher_guess_list)

    def calc_single_err(self, guess, teacher_guess):
        if not self.is_valid(guess):
            return 1

        guess = int(guess)
        teacher_guess = int(teacher_guess)
        return abs(guess - teacher_guess) / (self.max - self.min)

    def calc_err(self, guess_list, teacher_guess_list):
        err_list = [
            1 - self.calc_single_err(i, j) for i, j in zip(guess_list, teacher_guess_list)
        ]
        return sum(err_list) / len(err_list)

    def test_multi_times(self, model, times, teacher_forcing_mode="l0"):
        # teacher forcing options:
        # "l0": no teacher forcing, context is cleared after each test
        # "l1": naive teacher forcing, context is cleared after each test
        # "l2": no teacher forcing during the current test, previous context is used as initial prompt
        #       after forced
        # "l3": similar to "l4" but the final test runs in the "l0" mode
        # "l4": full teacher forcing, previous context is used as initial prompt after forced

        pass
