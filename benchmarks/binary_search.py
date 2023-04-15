import random
import re
from loguru import logger

from models import BSModel
from utils import DialogLogger, dict_mean


class BinarySearchEvaluator():
    def __init__(self, min=0, max=100, format_tolerant=False, max_retry=0, max_guess=None):
        assert min <= max
        self.min = min
        self.max = max
        self.format_tolerant = format_tolerant
        # `max_retry` and `max_guess` are only activated when not teacher forcing
        self.max_retry = max_retry
        self.max_guess = max_guess if max_guess is not None else self.max - self.min + 1
        self.dialog_logger = DialogLogger(order=["System", "Q", "A", "T"])
        self.teacher = BSModel(min, max)

    def reset(self):
        self.teacher.reset()
        self._teacher_qa_list = None
        self._target = None

    @property
    def init_prompt(self):
        #    "I will only give responses such as 'The true number is bigger than this guess' or 'The true number is smaller than this guess' or 'The true number is equal to this guess'. " \
        return "You are required to guess the random number which I have just picked between {} and {}. " \
               "I will only tell you whether the true number is bigger or lower than your guess." \
               "Adjust your guess according to my response. " \
               "Try as few times as you can. " \
               "Reply 'OK' if you understand. " \
               "You can only reply with a integer number between {} and {}.".format(self.min, self.max, self.min, self.max)

    def reset_model(self, model, init_prompt=None, verbose=True):
        if init_prompt is None:
            init_prompt = self.init_prompt

        if verbose:
            self.dialog_logger.info(System=init_prompt)

        model.reset(init_prompt + "\n\n")
        return

    def get_prompt(self, guess):
        if guess < self._target:
            return f"The true number is bigger than {guess}."
        if guess > self._target:
            return f"The true number is smaller than {guess}."

        return f"Right answer. The true number is equal to {guess}."

    def is_valid(self, guess):
        if self.format_tolerant:
            nums = re.findall(r'\d+', guess)
            if not len(nums):
                return False

            guess = int(nums[0])
            return self.min <= guess and guess <= self.max

        try:
            guess = int(guess)
            return self.min <= guess and guess <= self.max
        except ValueError:
            return False

    def calc_err(self, guess, target_guess):
        if not self.is_valid(guess):
            return 1

        if self.format_tolerant:
            guess = re.findall(r'\d+', guess)[0]
        guess = int(guess)
        target_guess = int(target_guess)
        return abs(guess - target_guess) / (self.max - self.min)

    def calc_metric(self, guess_list, target_guess_list):
        err_list = [
            self.calc_err(i, j) for i, j in zip(guess_list, target_guess_list)
        ]

        metrics = {
            "mean_err": sum(err_list) / len(err_list),
            "sum_err": sum(err_list),
            "min_err": min(err_list)
        }
        return metrics

    def refresh_teacher_qa(self):
        # teacher always recieve a fresh initial prompt without previous context
        self.reset_model(self.teacher, verbose=False)
        self._teacher_qa_list = []

        guess = None
        guess_list = []
        prompt = "START"

        while guess != self._target:
            guess = self.teacher(prompt)
            guess_list.append(guess)
            self._teacher_qa_list.append((prompt, guess))

            prompt = self.get_prompt(guess)

        self._teacher_qa_list.append((prompt, None))

    def _test_no_tf(self, model):
        guess = None
        guess_list = []
        prompt = "START"

        while guess != self._target:
            if len(guess_list) >= self.max_guess:
                logger.info(
                    f"Max guess times reached, stop guessing now."
                )
                return guess_list

            self.dialog_logger.info(Q=prompt)

            for _ in range(self.max_retry + 1):
                guess = model(prompt)
                self.dialog_logger.info(A=guess)

                if self.is_valid(guess):
                    break

                prompt = "Invalid reply. " \
                         "You can only reply with a integer number between " \
                         f"{self.min} and {self.max}."
                self.dialog_logger.info(Q=prompt)

            if not self.is_valid(guess):
                guess_list.append(guess)
                logger.info(
                    f"Max retry times reached, stop guessing now."
                )
                return guess_list

            if self.format_tolerant:
                guess_ = re.findall(r'\d+', guess)[0]

                if guess_ != guess:
                    logger.info(
                        f"Format tolerance enabled, force the model reply to {guess_}."
                    )
                    model.force(guess_)
                    guess = guess_

            guess_list.append(guess)
            guess = int(guess)
            prompt = self.get_prompt(guess)

        self.dialog_logger.info(Q=prompt)

        return guess_list

    def _test_tf(self, model):
        # no retry when teacher forcing
        guess = None
        guess_list = []
        teacher_guess_list = []

        self.refresh_teacher_qa()

        for prompt, teacher_guess in self._teacher_qa_list[:-1]:
            self.dialog_logger.info(Q=prompt)

            guess = model(prompt)

            model.force(teacher_guess)
            self.dialog_logger.info(A=guess, T=teacher_guess)

            guess_list.append(guess)
            teacher_guess_list.append(teacher_guess)

        self.dialog_logger.info(Q=self._teacher_qa_list[-1][0])

        return guess_list, teacher_guess_list

    def test_one_time(self, model, teacher_forcing=False, init_prompt=None):
        self.reset()
        self.reset_model(model, init_prompt)

        self._target = random.randint(self.min, self.max)
        logger.info("Picked Random Number: {}".format(self._target))

        if teacher_forcing:
            guess_list, teacher_guess_list = self._test_tf(model)
            return self.calc_metric(guess_list, teacher_guess_list)
        else:
            guess_list = self._test_no_tf(model)
            if not self.is_valid(guess_list[-1]):
                guess_list = guess_list[:-1]

            target_list = [self._target] * len(guess_list)
            return self.calc_metric(guess_list, target_list)

    def _independent_test(self, model, times, teacher_forcing):
        results = []
        for i in range(times):
            self.reset_model(model)
            result = self.test_one_time(model, teacher_forcing)
            logger.info(f"Evaluation result #{i}: {result}")
            results.append(result)

        return dict_mean(results)

    def _context_kept_test(self, model, times, teacher_forcing_mode):
        # previous context won't be printed again as the initial prompt
        # during the next text run

        def get_tf_flag(i):
            # i: current run is the `i`-th run. `i` in [0, `times` - 1]
            if teacher_forcing_mode == "l2":
                return False

            if teacher_forcing_mode == "l4":
                return True

            return i < times - 1

        composed_pre_ctx = self.init_prompt + "\nHere are some examples (the right answer for each example is different):\n"
        results = []
        for i in range(times):
            result = self.test_one_time(
                model, get_tf_flag(i),
                init_prompt=None if i == 0 else composed_pre_ctx
            )

            logger.info(f"Evaluation result #{i}: {result}")
            results.append(result)

            if not get_tf_flag(i):
                self.refresh_teacher_qa()

            example_ctx = f"Example #{i + 1}: \n" + model.rebuild_context(self._teacher_qa_list)
            composed_pre_ctx += example_ctx

        return dict_mean(results)

    def test_multi_times(self, model, times, teacher_forcing_mode="l0"):
        # teacher forcing options:
        # "l0": no teacher forcing, context is cleared after each test
        # "l1": naive teacher forcing, context is cleared after each test
        # "l2": no teacher forcing during the current test, previous context is used as
        #       initial prompt after forced
        # "l3": similar to "l4" but the final test runs in the "l2" mode
        # "l4": full teacher forcing, previous context is used as initial prompt after forced

        assert teacher_forcing_mode in ["l0", "l1", "l2", "l3", "l4"], teacher_forcing_mode

        if teacher_forcing_mode in ["l0", "l1"]:
            return self._independent_test(model, times, teacher_forcing_mode=="l0")

        return self._context_kept_test(model, times, teacher_forcing_mode)
