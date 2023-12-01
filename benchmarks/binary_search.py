import random
import re

from loguru import logger

from models import BSModel
from utils import Invalid, FormatInvalid, ValueInvalid
from .benchmark import Benchmark


class BinarySearchEvaluator(Benchmark):
    def __init__(
        self, min=0, max=100,
        format_tolerant=True, max_retry=0, max_step=None,
        verbose=True, output_dir=None, save_period=-1
    ):
        super(BinarySearchEvaluator, self).__init__(
            format_tolerant, max_retry, max_step, verbose, output_dir, save_period
        )
        assert min <= max
        self.min = min
        self.max = max
        self.teacher = BSModel(min, max)

    def reset(self, test_case=None):
        super(BinarySearchEvaluator, self).reset(test_case)

        if test_case is None:
            logger.info("Generating random number.")
            self._target = random.randint(self.min, self.max)
        else:
            logger.info("Using pre-generated random number.")
            self._target = test_case["target"]
            assert self.min <= self._target and self._target <= self.max, self._target

    @property
    def default_instruction(self):
        return "You are required to guess the random number which I have just picked between {} and {}. " \
               "I will only tell you whether the true number is bigger or lower than your guess. " \
               "Adjust your guess according to my response. " \
               "Try as few times as you can. " \
               "You can only reply with a integer number between {} and {}." \
               .format(self.min, self.max, self.min, self.max)

    def _get_prompt(self, guess):
        if guess < self._target:
            return f"The true number is bigger than {guess}."
        if guess > self._target:
            return f"The true number is smaller than {guess}."

        return f"Right answer. The true number is equal to {guess}."

    def _refresh_teacher_qa(self):
        super(BinarySearchEvaluator, self)._refresh_teacher_qa()

        guess = None
        prompt = "START"

        while guess != self._target:
            guess = int(self.teacher(prompt))
            self._teacher_qa_list.append((prompt, guess))

            prompt = self._get_prompt(guess)

        self._teacher_qa_list.append((prompt, None))

    def _extract_answer(self, reply):
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

    def _calc_err(self, guess, target):
        # calculate the error between a single guess and target
        if isinstance(guess, Invalid):
            return 1

        guess = int(guess)
        target = int(target)
        return abs(guess - target) / (self.max - self.min)

    def calc_metric_tf(self, answer_list, target_list):
        # calculate all the metrics given a list of guesses and targets
        if not len(answer_list):
            return {
                "avg_err": 1.0,
                "sum_err": 1.0,
                "min_err": 1.0,
            }

        err_list = [
            self._calc_err(i, j) for i, j in zip(answer_list, target_list)
        ]

        metrics = {
            "avg_err": sum(err_list) / len(err_list),
            "sum_err": sum(err_list),
            "min_err": min(err_list),
            "acc": sum([err == 0 for err in err_list]) / len(err_list)
        }

        return metrics

    def calc_metric_no_tf(self, answer_list, target_list):
        # Since we are interested in how close the model gets to the target number before
        # it quits, the final invalid guess is removed when calculating metrics
        if isinstance(answer_list[-1], Invalid):
            answer_list = answer_list[:-1]

        return self.calc_metric_tf(answer_list, target_list)

    def _test_no_tf(self, model):
        # test one time without teacher forcing
        answer = None
        answer_list = []
        prompt = "START"

        retry_cnt = 0

        while (
            answer != self._target
            # stop when reaching `self.max_step`
            and (self.max_step is None or len(answer_list) < self.max_step)
            # stop when reaching `self.max_retry`
            and retry_cnt < (self.max_retry + 1)
        ):
            self.dialog_logger.info(Q=prompt)

            reply = model(prompt)
            self.dialog_logger.info(A=reply)

            answer = self._extract_answer(reply)

            # if `reply` is formatted, force the new reply
            if not isinstance(answer, FormatInvalid) \
               and str(getattr(answer, "output", answer)) != reply:
                assert self.format_tolerant
                formatted = getattr(answer, "output", answer)
                assert isinstance(formatted, int)
                logger.info(f"Format tolerance enabled, force the model reply to {formatted}.")
                model.force(str(formatted))

            if not isinstance(answer, Invalid):
                prompt = self._get_prompt(answer)
                answer_list.append(answer)
                retry_cnt = 0
                continue

            if retry_cnt == 0:
                prompt = "Invalid reply. You can only reply with a integer number between " \
                        f"{self.min} and {self.max}. Try again." + prompt
            retry_cnt += 1

        self.dialog_logger.info(Q=prompt)

        if isinstance(answer, Invalid):
            answer_list.append(answer)  # save the last invalid
            logger.info("Max retry times reached, stop interaction now.")
        elif answer != self._target:  # target not achieved
            logger.info("Max steps reached, stop the interaction now.")

        return answer_list

    def _test_tf(self, model):
        # test one time with teacher forcing
        answer = None
        answer_list = []
        teacher_answer_list = []

        self._refresh_teacher_qa()

        # no retry when teacher forcing
        for prompt, teacher_answer in self._teacher_qa_list[:-1]:
            self.dialog_logger.info(Q=prompt)

            reply = model(prompt)

            model.force(str(teacher_answer))
            self.dialog_logger.info(A=reply, T=teacher_answer)

            answer = self._extract_answer(reply)

            answer_list.append(answer)
            teacher_answer_list.append(teacher_answer)

        self.dialog_logger.info(Q=self._teacher_qa_list[-1][0])

        return answer_list, teacher_answer_list

    def naive_test(self, model, teacher_forcing=False, instruction=None, test_case=None, example_qa_lists=None):
        super(BinarySearchEvaluator, self).naive_test(
            model, teacher_forcing, instruction, test_case, example_qa_lists
        )

        logger.info("Target number: {}".format(self._target))

        if teacher_forcing:
            answer_list, teacher_answer_list = self._test_tf(model)
            metric = self.calc_metric_tf(answer_list, teacher_answer_list)
        else:
            answer_list = self._test_no_tf(model)
            teacher_answer_list = []

            target_list = [self._target] * len(answer_list)
            metric = self.calc_metric_no_tf(answer_list, target_list)

        result = self._get_result(
            metric, answer_list, teacher_answer_list,
            model.history, teacher_forcing, instruction
        )

        return metric, result

    def _get_result(
        self, metric, answer_list, teacher_answer_list,
        model_history, teacher_forcing, instruction=None
    ):
        result = super(BinarySearchEvaluator, self)._get_result(
            metric, answer_list, teacher_answer_list, model_history, teacher_forcing, instruction
        )

        result["env"].update(
            min=self.min,
            max=self.max,
            target=self._target,
        )

        return result

    def _pack_results(self, single_results, teacher_forcing_mode):
        metric, full_result = super(BinarySearchEvaluator, self)._pack_results(
            single_results, teacher_forcing_mode
        )

        full_result["env"]["min"] = self.min
        full_result["env"]["max"] = self.max

        return metric, full_result
