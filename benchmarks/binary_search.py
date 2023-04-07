import random
from loguru import logger

from models import BSModel
from utils import DialogLogger


class BinarySearchEvaluator():
    def __init__(self, min=0, max=100):
        self.min = min
        self.max = max
        self.dialog_logger = DialogLogger(order=["Q", "A", "T"])
        self.teacher = BSModel(min, max)

    def reset(self):
        self.teacher.reset()
        self._teacher_qa_list = None
        self._target = None

    @property
    def init_prompt(self):
        return "You are required to guess the random number which I have just picked between {} and {}.\n" \
               "I will only give responses such as 'The true number is bigger than this guess' or 'The true number is smaller than this guess' or 'The true number is equal to this guess'.\n" \
               "Adjust your guess according to my response.\n" \
               "Try as few times as you can.\n" \
               "Start guessing after receiving 'START' command.\n" \
               "Stop guessing after receiving 'STOP' command.\n" \
               "Reply 'OK' if you understand.".format(self.min, self.max)

    def init_model(self, model, verbose=True):
        if verbose:
            self.dialog_logger.info(Q=self.init_prompt)

        reply = model(self.init_prompt)

        if verbose:
            self.dialog_logger.info(A=reply)
        return reply.strip() == "OK"

    def get_prompt(self, guess):
        if guess < self._target:
            return "The true number is bigger than this guess"
        if guess > self._target:
            return "The true number is smaller than this guess"

        return "The true number is equal to this guess"

    def is_valid(self, guess):
        try:
            guess = int(guess)
            return self.min <= guess and guess <= self.max
        except ValueError:
            return False

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

    def refresh_teacher_qa(self):
        # teacher always recieve a fresh initial prompt without previous context
        self.init_model(self.teacher, verbose=False)
        self._teacher_qa_list = [(self.init_prompt, "OK")]

        guess = None
        guess_list = []
        prompt = "START"

        while guess != self._target:
            guess = self.teacher(prompt)
            guess_list.append(guess)
            self._teacher_qa_list.append((prompt, guess))

            prompt = self.get_prompt(guess)

        self._teacher_qa_list.append((prompt, ""))

    def _test_no_tf(self, model):
        if not self.init_model(model):
            raise ValueError("Invalid Reply")

        guess = None
        guess_list = []
        prompt = "START"

        while guess != self._target:
            self.dialog_logger.info(Q=prompt)

            guess = model(prompt)
            guess_list.append(guess)

            if not self.is_valid(guess):
                raise ValueError(f"Invalid Reply: {guess}")

            self.dialog_logger.info(A=guess)

            prompt = self.get_prompt(guess)

        self.dialog_logger.info(Q=prompt)

        return len(guess_list)

    def _test_tf(self, model):
        guess = None
        guess_list = []
        teacher_guess_list = []

        self.refresh_teacher_qa()

        for prompt, teacher_guess in self._teacher_qa_list[:-1]:
            self.dialog_logger.info(Q=prompt)

            guess = model(prompt)
            model.teacher_force(teacher_guess)
            guess_list.append(guess)
            teacher_guess_list.append(teacher_guess)

            self.dialog_logger.info(A=guess, T=teacher_guess)

        self.dialog_logger.info(Q=self._teacher_qa_list[-1][0])

        return self.calc_err(guess_list, teacher_guess_list)

    def test_one_time(self, model, teacher_forcing=False):
        self.reset()

        self._target = random.randint(self.min, self.max)
        logger.info("Picked Random Number: {}".format(self._target))

        if teacher_forcing:
            return self._test_tf(model)
        else:
            return self._test_no_tf(model)

    def _independent_test(self, model, times, teacher_forcing):
        results = []
        for _ in range(times):
            model.reset()
            results.append(self.test_one_time(model, teacher_forcing))

        # TODO: figure out how to merge results from completed runs and failed ones into one metric
        raise NotImplementedError

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

        composed_pre_ctx = ""
        results = []
        for i in range(times):
            model.reset(init_context=composed_pre_ctx)
            results.append(self.test_one_time(model, get_tf_flag(i)))

            if not get_tf_flag(i):
                self.refresh_teacher_qa()

            composed_pre_ctx += model.rebuild_context(self._teacher_qa_list)

        # TODO: figure out how to merge results into one metric
        raise NotImplementedError

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
            return self._independent_test(model, times, teacher_forcing_mode == "l1")

        return self._context_kept_test(model, times, teacher_forcing_mode)
