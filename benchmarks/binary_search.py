import random
import re

from loguru import logger

from models import BSModel
from utils import DialogLogger, Invalid, FormatInvalid, ValueInvalid, dict_mean


class BinarySearchEvaluator():
    def __init__(self, min=0, max=100, format_tolerant=True, max_retry=0, max_step=None):
        assert min <= max
        self.min = min
        self.max = max
        self.format_tolerant = format_tolerant
        # `max_retry` and `max_step` are only activated when not teacher forcing
        self.max_retry = max_retry
        self.max_step = max_step if max_step is not None else self.max - self.min + 1
        self.teacher = BSModel(min, max)
        self.dialog_logger = DialogLogger(order=["System", "Q", "A", "T"])

    def reset(self):
        self.teacher.reset()
        self._teacher_qa_list = None
        self._target = None

    @property
    def default_insturction(self):
        return "You are required to guess the random number which I have just picked between {} and {}. " \
               "I will only tell you whether the true number is bigger or lower than your guess. " \
               "Adjust your guess according to my response. " \
               "Try as few times as you can. " \
               "You can only reply with a integer number between {} and {}." \
               .format(self.min, self.max, self.min, self.max)

    def reset_model(self, model, instruction=None, verbose=True):
        # clear dialog history and give instruction
        # will use `self.default_insturction` if `instruction` is None
        if instruction is None:
            instruction = self.default_insturction

        if verbose:
            self.dialog_logger.info(System=instruction)

        model.reset(instruction)
        return

    def _get_prompt(self, guess):
        if guess < self._target:
            return f"The true number is bigger than {guess}."
        if guess > self._target:
            return f"The true number is smaller than {guess}."

        return f"Right answer. The true number is equal to {guess}."

    def extract_answer(self, reply):
        # parse reply from model and return the formatted answer
        # return an `Invalid` if failed to do so
        if self.format_tolerant:
            nums = re.findall(r'\d+', reply)
            if not len(nums):
                return FormatInvalid(reply)

            guess = int(nums[0])

            if guess < self.min or guess > self.max:
                return ValueInvalid(guess)
            return guess

        try:
            guess = int(reply)

            if guess < self.min or guess > self.max:
                return ValueInvalid(guess)
            return guess
        except ValueError:
            return FormatInvalid(guess)

    def calc_err(self, guess, target):
        # calculate the error between a single guess and target
        if isinstance(guess, Invalid):
            return 1

        guess = int(guess)
        target = int(target)
        return abs(guess - target) / (self.max - self.min)

    def calc_metric(self, guess_list, target_list):
        # calculate all the metrics given a list of guesses and targets
        if not len(guess_list):
            return {
                "avg_err": 1.0,
                "sum_err": 1.0,
                "min_err": 1.0,
            }

        err_list = [
            self.calc_err(i, j) for i, j in zip(guess_list, target_list)
        ]

        metrics = {
            "avg_err": sum(err_list) / len(err_list),
            "sum_err": sum(err_list),
            "min_err": min(err_list),
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

            prompt = self._get_prompt(guess)

        self._teacher_qa_list.append((prompt, None))

    def _test_no_tf(self, model):
        # test one time without teacher forcing
        guess = None
        guess_list = []
        prompt = "START"

        while guess != self._target:
            if len(guess_list) >= self.max_step:
                logger.info("Max guess times reached, stop guessing now.")
                return guess_list

            self.dialog_logger.info(Q=prompt)

            for _ in range(self.max_retry + 1):
                reply = model(prompt)
                self.dialog_logger.info(A=reply)

                guess = self.extract_answer(reply)

                if not isinstance(guess, Invalid):
                    break

                # if `reply` is formatted, force the new reply
                if self.format_tolerant and isinstance(guess, ValueInvalid):
                    formatted = guess.output
                    assert isinstance(formatted, int)
                    logger.info(f"Format tolerance enabled, force the model reply to {formatted}.")
                    model.force(str(formatted))

                prompt = "Invalid reply. " \
                         "You can only reply with a integer number between " \
                         f"{self.min} and {self.max}. Try again."
                self.dialog_logger.info(Q=prompt)

            if isinstance(guess, Invalid):
                guess_list.append(guess)
                logger.info("Max retry times reached, stop guessing now.")
                return guess_list

            # if the final guess is valid due to `self.format_tolerant`, force the reply
            if str(guess) != reply:
                assert self.format_tolerant, "Reply is changed with format tolerance disabled"
                logger.info(f"Format tolerance enabled, force the model reply to {guess}.")
                model.force(str(guess))

            guess_list.append(guess)
            prompt = self._get_prompt(guess)

        self.dialog_logger.info(Q=prompt)

        return guess_list

    def _test_tf(self, model):
        # test one time with teacher forcing
        guess = None
        guess_list = []
        teacher_guess_list = []

        self.refresh_teacher_qa()

        # no retry when teacher forcing
        for prompt, teacher_guess in self._teacher_qa_list[:-1]:
            self.dialog_logger.info(Q=prompt)

            reply = model(prompt)

            model.force(str(teacher_guess))
            self.dialog_logger.info(A=reply, T=teacher_guess)

            guess = self.extract_answer(reply)

            guess_list.append(guess)
            teacher_guess_list.append(teacher_guess)

        self.dialog_logger.info(Q=self._teacher_qa_list[-1][0])

        return guess_list, teacher_guess_list

    def test_one_time(self, model, teacher_forcing=False, instruction=None):
        self.reset()
        # will use `self.default_insturction` if `instruction` is None
        self.reset_model(model, instruction)

        self._target = random.randint(self.min, self.max)
        logger.info("Picked Random Number: {}".format(self._target))

        if teacher_forcing:
            guess_list, teacher_guess_list = self._test_tf(model)
            metric = self.calc_metric(guess_list, teacher_guess_list)
        else:
            guess_list = self._test_no_tf(model)
            # Since we are interested in how close the model gets to the target number before
            # it quits, the final invalid guess is removed when calculating metrics
            if isinstance(guess_list[-1], Invalid):
                guess_list = guess_list[:-1]

            target_list = [self._target] * len(guess_list)
            metric = self.calc_metric(guess_list, target_list)

        full_result = {}
        full_result["metric"] = metric
        full_result["output"] = dict(
            guess_list=guess_list,
            teacher_guess_list=teacher_guess_list if teacher_forcing else None
        )
        full_result["env"] = dict(
            min=self.min,
            max=self.max,
            target=self._target,
            teacher_forcing=teacher_forcing,
            instruction=self.default_insturction if instruction is None else instruction
        )
        full_result["history"] = dict(
            model_history=model.history,
            teacher_history=self._teacher_qa_list if teacher_forcing else None
        )

        return metric, full_result

    def _pack_multi_time_result(self, metrics, single_results, teacher_forcing_mode):
        # summarize the metrics from each test run and pack the detailed results
        metric = dict_mean(metrics)

        metric = {"mean_" + k: v for k, v in metric.items()}

        full_result = {}
        full_result["metric"] = metric
        full_result["env"] = dict(
            min=self.min,
            max=self.max,
            times=len(metrics),
            teacher_forcing_mode=teacher_forcing_mode,
            default_insturction=self.default_insturction
        )
        full_result["single_results"] = single_results

        return metric, full_result

    def _independent_test(self, model, times, teacher_forcing_mode):
        teacher_forcing = teacher_forcing_mode == "l1"

        metrics = []
        single_results = []
        for i in range(times):
            metric, single_result = self.test_one_time(model, teacher_forcing)

            logger.info(f"Evaluation metric #{i}: {metric}")
            metrics.append(metric)
            single_results.append(single_result)

        return self._pack_multi_time_result(metrics, single_results, teacher_forcing_mode)

    def _context_kept_test(self, model, times, teacher_forcing_mode):
        # model's history will be cleared before each run
        # teacher model's history will be used as example in the intruction prompt
        # at the beginning of each run

        def get_tf_flag(i):
            # i: current run is the `i`-th run. `i` in [0, `times` - 1]
            if teacher_forcing_mode == "l2":
                return False

            if teacher_forcing_mode == "l4":
                return True

            return i < times - 1

        instruction_w_examples = self.default_insturction \
            + "\nHere are some examples (the right answer for each example is different):\n"

        metrics = []
        single_results = []
        for i in range(times):
            metric, single_result = self.test_one_time(
                model, get_tf_flag(i),
                instruction=None if i == 0 else instruction_w_examples
            )

            logger.info(f"Evaluation metric #{i}: {metric}")

            if not get_tf_flag(i):
                self.refresh_teacher_qa()
                single_result["history"]["teacher_history"] = self._teacher_qa_list

            metrics.append(metric)
            single_results.append(single_result)

            example_ctx = f"Example #{i + 1}: \n" + model.rebuild_context(self._teacher_qa_list)
            instruction_w_examples += example_ctx

        return self._pack_multi_time_result(metrics, single_results, teacher_forcing_mode)

    def test_multi_time(self, model, times, teacher_forcing_mode="l0"):
        # teacher forcing options:
        # "l0": no teacher forcing, context is cleared after each test
        # "l1": naive teacher forcing, context is cleared after each test
        # "l2": no teacher forcing during the current test, previous context is used as
        #       initial prompt after forced
        # "l3": similar to "l4" but the final test runs in the "l2" mode
        # "l4": full teacher forcing, previous context is used as initial prompt after forced

        assert teacher_forcing_mode in ["l0", "l1", "l2", "l3", "l4"], teacher_forcing_mode

        if teacher_forcing_mode in ["l0", "l1"]:
            return self._independent_test(model, times, teacher_forcing_mode)

        return self._context_kept_test(model, times, teacher_forcing_mode)
