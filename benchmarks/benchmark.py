import abc
from loguru import logger
import json
import os.path as osp

from utils import DialogLogger, dict_mean, setup_logger


class Benchmark(metaclass=abc.ABCMeta):
    def __init__(
            self, format_tolerant=True, max_retry=0, max_step=None,
            verbose=True, output_dir=None
        ):
        self.format_tolerant = format_tolerant
        # `max_retry` and `max_step` are only activated when not teacher forcing
        self.max_retry = max_retry
        # `max_step` will be deactivated if `max_step is None`
        self.max_step = max_step

        self.teacher = None
        self.dialog_logger = DialogLogger(order=["System", "Q", "A", "T"], enabled=verbose)
        self.test_cases = []
        if output_dir:
            setup_logger(filename=osp.join(output_dir, "log.txt"))

    def load_testcases_from_file(self, path):
        self.test_cases = json.load(open(path))
        assert isinstance(self.test_cases, list), self.test_cases

    def reset(self, test_case):
        self.test_case = test_case
        self.teacher.reset()
        self._teacher_qa_list = []

    def reset_model(self, model, instruction=None, verbose=True):
        # clear dialog history and give instruction
        # will use `self.default_instruction` if `instruction` is None
        if instruction is None:
            instruction = self.default_instruction

        if verbose:
            self.dialog_logger.info(System=instruction)

        model.reset(instruction)

    def naive_test(self, model, teacher_forcing=False, instruction=None, test_case=None):
        self.reset(test_case)
        # will use `self.default_instruction` if `instruction` is None
        self.reset_model(model, instruction)

    def _refresh_teacher_qa(self):
        # teacher always recieve a fresh initial prompt without previous context
        self.reset_model(self.teacher, verbose=False)
        self._teacher_qa_list = []

    def _get_result(
        self, metric, answer_list, teacher_answer_list,
        model_history, teacher_forcing, instruction=None
    ):
        assert teacher_forcing or len(teacher_answer_list) == 0, \
            "`teacher_answer_list` must be empty when teacher forcing is disabled.\n" \
            f"teacher_answer_list={teacher_answer_list}"

        assert teacher_forcing or len(self._teacher_qa_list) == 0, \
            "`self._teacher_qa_list` must be empty when teacher forcing is disabled.\n" \
            f"self._teacher_qa_list={self._teacher_qa_list}"

        result = {}
        result["metric"] = metric
        result["output"] = dict(
            answer_list=answer_list,
            teacher_answer_list=teacher_answer_list
        )
        result["env"] = dict(
            teacher_forcing=teacher_forcing,
            instruction=self.default_instruction if instruction is None else instruction
        )
        result["history"] = dict(
            model_history=model_history,
            teacher_history=self._teacher_qa_list
        )

        return result

    def _pack_results(self, metrics, single_results, teacher_forcing_mode):
        # summarize the metrics from each test run and pack the detailed results
        metric = dict_mean(metrics)

        metric = {"mean_" + k: v for k, v in metric.items()}

        full_result = {}
        full_result["metric"] = metric
        full_result["env"] = dict(
            times=len(metrics),
            teacher_forcing_mode=teacher_forcing_mode,
            default_instruction=self.default_instruction
        )
        full_result["single_results"] = single_results

        return metric, full_result

    def _independent_test(self, model, times, teacher_forcing_mode):
        teacher_forcing = teacher_forcing_mode == "l1"

        metrics = []
        single_results = []
        for i in range(times):
            metric, single_result = self.naive_test(model, teacher_forcing)

            logger.info(f"Evaluation metric #{i}: {metric}")
            metrics.append(metric)
            single_results.append(single_result)

        return self._pack_results(metrics, single_results, teacher_forcing_mode)

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

        instruction_w_examples = self.default_instruction \
            + "\nHere are some examples (the right answer for each example is different):\n"

        metrics = []
        single_results = []
        for i in range(times):
            metric, single_result = self.naive_test(
                model, get_tf_flag(i),
                instruction=None if i == 0 else instruction_w_examples
            )

            logger.info(f"Evaluation metric #{i}: {metric}")

            if not get_tf_flag(i):
                self._refresh_teacher_qa()
                single_result["history"]["teacher_history"] = self._teacher_qa_list

            metrics.append(metric)
            single_results.append(single_result)

            example_ctx = f"Example #{i + 1}: \n" + model.rebuild_context(self._teacher_qa_list)
            instruction_w_examples += example_ctx

        return self._pack_results(metrics, single_results, teacher_forcing_mode)

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

    def _add_examples(self, instruction, qa_lists, rebuild_context_func):
        if not qa_lists:
            return instruction

        instruction_w_examples = instruction \
            + "\nHere are some examples (the right answer for each example is different):\n"

        for i, qa_list in enumerate(qa_lists):
            example_ctx = f"Example #{i + 1}: \n" + rebuild_context_func(qa_list)
            instruction_w_examples += example_ctx

        return instruction_w_examples

    def _init_teacher_qa_lists(self, num_examples):
        if not num_examples:
            return []

        teacher_qa_lists = []
        for example_case in self.test_cases[- num_examples:]:
            self.reset(example_case)
            self._refresh_teacher_qa()
            teacher_qa_lists.append(self._teacher_qa_list)

        return teacher_qa_lists

    def test_with_examples(self, model, times, num_examples=0, teacher_forcing=False):
        assert times <= len(self.test_cases), self.test_cases
        assert num_examples <= len(self.test_cases), self.test_cases

        teacher_qa_lists = self._init_teacher_qa_lists(num_examples)
        metrics = []
        single_results = []

        for test_case in self.test_cases[:times]:
            instruction_w_examples = self._add_examples(
                self.default_instruction, teacher_qa_lists, model.rebuild_context
            )
            metric, single_result = self.naive_test(
                model, teacher_forcing,
                instruction=instruction_w_examples,
                test_case=test_case
            )
            logger.info(f"Evaluation metric #{i}: {metric}")

            if num_examples:
                teacher_qa_lists = teacher_qa_lists[1:]
                if not teacher_forcing:
                    self._refresh_teacher_qa()
                    # single_result["history"]["teacher_history"] = deepcopy(self._teacher_qa_list)
                teacher_qa_lists.append(self._teacher_qa_list)

            metrics.append(metric)
            single_results.append(single_result)

        return self._pack_results(metrics, single_results, teacher_forcing_mode=teacher_forcing)

    @property
    @abc.abstractmethod
    def default_instruction(self):
        pass

    @abc.abstractmethod
    def _get_prompt(self):
        pass

    @abc.abstractmethod
    def _extract_answer(self):
        """
            return extracted answer with the correct format after checking for
            errors such as invalid format or value.
            return `Invalid` accordingly if erros are found.
        """
        pass

    @abc.abstractmethod
    def calc_metric_tf(self, answer_list, target_list):
        pass

    @abc.abstractmethod
    def calc_metric_no_tf(self, answer_list, target_list):
        pass

    @abc.abstractmethod
    def _test_no_tf(self, model):
        pass

    @abc.abstractmethod
    def _test_tf(self, model):
        pass
