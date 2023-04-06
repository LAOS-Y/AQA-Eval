import random
from loguru import logger

from utils import DialogLogger


class BinarySearchEvaluator():
    def __init__(self, min=0, max=100):
        self.min = min
        self.max = max
        self.dialog_logger = DialogLogger(order=["Q", "A", "T"])

    def init_model(self, model, teacher_forcing=False):
        prompt = "You are required to guess the random number which I have just picked between {} and {}.\n" \
                 "I will only give responses such as 'The true number is bigger than this guess' or 'The true number is smaller than this guess' or 'The true number is equal to this guess'.\n" \
                 "Adjust your guess according to my response.\n" \
                 "Try as few times as you can.\n" \
                 "Start guessing after receiving 'START' command.\n" \
                 "Stop guessing after receiving 'STOP' command.\n" \
                 "Reply 'OK' if you understand.".format(self.min, self.max)
        self.dialog_logger.info(Q=prompt)

        reply = model(prompt)
        if teacher_forcing:
            self.dialog_logger.info(A=reply, T="OK")
            model.teacher_force("OK")
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

    def test_one_time(self, model, teacher_forcing=False):
        if not self.init_model(model, teacher_forcing):
            raise ValueError("Invalid Reply")


        guess = None
        guess_list = []
        teacher_guess_list = []
        target = random.randint(self.min, self.max)
        prompt = "START"

        l, r = self.min, self.max

        logger.info("Picked Random Number: {}".format(target))

        while guess != target:
            self.dialog_logger.info(Q=prompt)

            guess = model(prompt)
            guess_list.append(guess)

            if not teacher_forcing and not self.is_valid(guess):
                raise ValueError(f"Invalid Reply: {guess}")

            if not teacher_forcing:
                self.dialog_logger.info(A=guess)
            else:
                old_guess = guess

                guess = (l + r) // 2
                teacher_guess_list.append(guess)
                model.teacher_force(guess)

                if target < guess:
                    r = guess
                else:
                    l = guess

                self.dialog_logger.info(A=old_guess, T=guess)

            prompt = self.get_prompt(guess, target)

        if teacher_forcing:
            return self.calc_err(guess_list, teacher_guess_list)

        return len(guess_list)

    def calc_single_err(self, guess, teacher_guess):
        if not self.is_valid(guess):
            return 1

        guess = int(guess)
        return abs(guess - teacher_guess) / (self.max - self.min)

    def calc_err(self, guess_list, teacher_guess_list):
        err_list = [
            1 - self.calc_single_err(i, j) for i, j in zip(guess_list, teacher_guess_list)
        ]
        return sum(err_list) / len(err_list)
