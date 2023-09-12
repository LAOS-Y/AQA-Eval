from copy import deepcopy
import networkx
import re

from loguru import logger

from models import BFSModel
from utils import DialogLogger, Invalid, FormatInvalid, ValueInvalid, dict_mean


# TODO: refine or just remove mcq and provide_state
class BFSEvaluator():
    def __init__(
        self, node_num=4, explain_algo=True, mcq=True, provide_state=True,
        format_tolerant=True, max_retry=0, max_step=20
    ):
        self.node_num = node_num
        self.mcq = mcq
        self.explain_algo = explain_algo
        self.provide_state = provide_state
        self.format_tolerant = format_tolerant
        # `max_retry` and `max_step` are only activated when not teacher forcing
        self.max_retry = max_retry
        self.max_step = max_step
        self.teacher = BFSModel()
        self.dialog_logger = DialogLogger(order=["System", "Q", "A", "T"])

    def reset(self):
        self._teacher_qa_list = []
        self.teacher.reset("")

    @property
    def default_insturction(self):
        instruction = "You are required to visit all the nodes in an undirected non-cyclic graph." \
                      "An undirected non-cyclic garph contains a set of node, and a set of edges that each connects a pair of nodes. " \
                      "Every time you visit a node, you will be given the adjacent nodes connected to this node. " \
                      "You can only visit nodes that are adjacent to the already visited nodes. " \
                      "You can only reply with a integer number indicating which node to be visited next. " \
                      "Please traverse the entire graph in as few rounds as possible." \
                      "Initially, you have already visited node 0." \

        if self.explain_algo:
            instruction += "You should use breadth first search algorithm. " \
                           "The algorithm works as follows:\n" \
                           "1. Initialize a queue data structure and add the starting node to the queue.\n" \
                           "2. While the queue is not empty, visit the first node and remove it from the queue.\n" \
                           "3. For nodes adjacent to the removed vertex, add the unvisited ones to the queue.\n" \
                           "4. Repeat steps 2-3 until the queue is empty."

        return instruction

    def reset_model(self, model, instruction=None, verbose=True):
        # clear dialog history and give instruction
        # will use `self.default_insturction` if `instruction` is None
        if instruction is None:
            instruction = self.default_insturction

        if verbose:
            self.dialog_logger.info(System=instruction)

        model.reset(instruction)

    def _get_adj_nodes(self, curr_node):
        return [n for _, n in self._graph.edges(curr_node)]

    def _get_valid_nodes(self, next_node, visited_nodes):
        valid_nodes = set(
            sum(
                [(self._get_adj_nodes(node) + [node]) for node in visited_nodes + [next_node]],
                start=[]
            )
        )
        assert self._start_node in valid_nodes

        return valid_nodes

    def _get_prompt(self, next_node, visited_nodes):
        '''
        Generate prompt used in exploration step

        Return: prompt (string)
        '''

        if len(set(visited_nodes + [next_node])) == len(self._graph.nodes):
            return "Well Done. You have visited all the nodes in the graph. " \
                   "Total number of steps: {}".format(len(visited_nodes[1:] + [next_node]))

        adj_nodes = self._get_adj_nodes(next_node)

        prompt = "Adjacent nodes: {}.".format(", ".join([str(i) for i in adj_nodes]))

        if self.provide_state:
            unvisited_adj_nodes = set(adj_nodes).difference(set(visited_nodes))
            if len(unvisited_adj_nodes) == 0:
                prompt += " You have visited all nodes adjacent to this node."
            else:
                prompt += " You have not visited node {}." \
                          .format(", ".join([str(i) for i in unvisited_adj_nodes]))
        if self.mcq:
            valid_nodes = self._get_valid_nodes(next_node, visited_nodes)

            prompt += " Choose the next node to visit: {}.".format(", ".join(valid_nodes))

        return prompt

    def extract_answer(self, reply, valid_nodes):
        # parse reply from model and return the formated answer
        # return an `Invalid` if failed to do so
        if self.format_tolerant:
            nums = re.findall(r'\d+', reply)
            if not len(nums):
                return FormatInvalid(reply)

            next_node = int(nums[0])

            if next_node not in valid_nodes:
                return ValueInvalid(next_node)
            return next_node

        try:
            next_node = int(reply)

            if next_node not in valid_nodes:
                return ValueInvalid(next_node)
            return next_node
        except ValueError:
            return FormatInvalid(next_node)

    def _init_queues(self):
        return self._get_adj_nodes(self._start_node), []

    def _update_queues(self, next_node, old_new_queues, visited_nodes):
        # doesn't change queues in-place
        assert isinstance(old_new_queues, tuple) and len(old_new_queues) == 2, old_new_queues
        old_queue, new_queue = deepcopy(old_new_queues)
        assert next_node in old_queue
        old_queue.pop(old_queue.index(next_node))
        assert next_node not in old_queue

        new_queue += [
            node
            for node in self._get_adj_nodes(next_node)
            if node not in (visited_nodes + new_queue)
        ]

        if not old_queue:
            old_queue = new_queue
            new_queue = []

        return old_queue, new_queue

    def _check_bfs(self, next_node, old_new_queues, visited_nodes):
        '''
        Check whether `next_node` follows BFS
        Will assume the previous steps in `node_history` already follow BFS

        Return
        - boolean: if selected interface follows bfs
        '''

        assert isinstance(old_new_queues, tuple) and len(old_new_queues) == 2, old_new_queues

        if isinstance(next_node, Invalid):
            return False

        old_queue, _ = old_new_queues
        assert old_queue, old_queue

        return next_node in old_queue

    _init_stack_or_queue = _init_queues
    _update_stack_or_queue = _update_queues
    _check_algo = _check_bfs

    def calc_decoverage(self, visited_nodes):
        assert self._start_node in visited_nodes
        return 1 - len(set(visited_nodes)) / len(self._graph.nodes)

    def refresh_teacher_qa(self):
        self.reset_model(self.teacher, verbose=False)
        self._teacher_qa_list = []

        response = ""
        prompt = self._get_prompt(self._start_node, [])
        decov_sum = self.calc_decoverage([self._start_node])

        next_node = self._start_node
        node_history = [self._start_node]

        # while exist node not visited
        while len(set(self._graph.nodes).difference(set(node_history))) != 0:
            response = self.teacher(prompt)
            next_node = int(response)

            self._teacher_qa_list.append((prompt, next_node))
            prompt = self._get_prompt(next_node, node_history)

            decov_sum += self.calc_decoverage(node_history + [next_node])
            node_history.append(next_node)

        self._teacher_qa_list.append((prompt, None))

        # remove start node in node_history
        node_history = node_history[1:]

        return decov_sum

    def calc_metric_no_tf(self, node_history):
        assert len(node_history) > 0
        assert node_history[0] != self._start_node

        decov_list = [self.calc_decoverage([self._start_node])]
        highest_cnt = 0
        check_algo_flag = True

        stack_or_queue = self._init_stack_or_queue()

        for idx, node in enumerate(node_history):  # remove the starting node
            if isinstance(node, Invalid):
                assert idx == len(node_history) - 1, \
                    f"Only the last node can be Invalid without teacher forcing. {node_history}"
                break

            # `check_algo_flag` will remain `True` until `model` stops following bfs
            if check_algo_flag:
                check_algo_flag = self._check_algo(
                    node, stack_or_queue, [self._start_node] + node_history[:idx]
                )
            if check_algo_flag:
                highest_cnt = idx + 1
                stack_or_queue = self._update_stack_or_queue(
                    node, stack_or_queue, [self._start_node] + node_history[:idx]
                )

            decov = self.calc_decoverage([self._start_node] + node_history[:idx + 1])
            assert decov <= decov_list[-1], "`decov_list` should be a non-ascent sequence"
            decov_list.append(decov)

        acc = highest_cnt / len(node_history)  # ignore the starting node  # dont ignore last invalid
        min_decov = decov_list[-1]
        sum_decov = sum(decov_list)

        metrics = {"acc": acc, "min_decov": min_decov, "sum_decov": sum_decov, "decov_list": decov_list}
        return metrics

    def calc_metric_tf(self, node_history, teacher_node_history):
        assert len(node_history) > 0
        assert node_history[0] != self._start_node

        decov_list = [self.calc_decoverage([self._start_node])]
        cnt = 0

        stack_or_queue = self._init_stack_or_queue()

        for idx, (node, teacher_node) in enumerate(
            zip(node_history, teacher_node_history)
        ):
            check_algo_flag = self._check_algo(
                node, stack_or_queue, [self._start_node] + teacher_node_history[:idx]
            )

            if check_algo_flag:
                cnt += 1

            stack_or_queue = self._update_stack_or_queue(
                teacher_node, stack_or_queue, [self._start_node] + teacher_node_history[:idx]
            )

            if isinstance(node, Invalid):
                decov = decov_list[-1]
            else:
                decov = self.calc_decoverage(
                    [self._start_node] + teacher_node_history[:idx] + [node]
                )
            assert decov <= decov_list[-1], "`decov_list` should be a non-ascent sequence"
            decov_list.append(decov)

        acc = cnt / len(node_history)  # ignore the starting node  # dont ignore last invalid
        min_decov = decov_list[-1]
        sum_decov = sum(decov_list)

        metrics = {"acc": acc, "min_decov": min_decov, "sum_decov": sum_decov, "decov_list": decov_list}
        return metrics

    def _test_no_tf(self, model):
        '''
        Return:
        - accuracy: percentage of node selected following bfs
        - decov_list: list of (1 - coverages)
        - trace of node explored by model
        '''
        prompt = self._get_prompt(self._start_node, [])
        node_history = []

        retry_cnt = 0

        valid_nodes = self._get_valid_nodes(self._start_node, [])

        while (
            len(set([self._start_node] + node_history)) != len(self._graph.nodes) and
            (len(node_history)) < self.max_step and retry_cnt < (self.max_retry + 1)
        ):
            self.dialog_logger.info(Q=prompt)

            reply = model(prompt)
            self.dialog_logger.info(A=reply)

            # start processing response in this iteration
            next_node = self.extract_answer(reply, valid_nodes)

            # if `reply` is formatted, force the new reply
            if not isinstance(next_node, FormatInvalid) \
               and str(getattr(next_node, "output", next_node)) != reply:
                assert self.format_tolerant
                formatted = str(getattr(next_node, "output", next_node))
                logger.info(f"Format tolerance enabled, force the model reply to {formatted}.")
                model.force(formatted)

            if not isinstance(next_node, Invalid):
                valid_nodes = self._get_valid_nodes(next_node, [self._start_node] + node_history)
                prompt = self._get_prompt(next_node, [self._start_node] + node_history)
                node_history.append(next_node)
                retry_cnt = 0

                continue

            if retry_cnt == 0:
                # TODO: maybe add mcq here?
                prompt = "Invalid reply. Try again. You can only reply with a " \
                         "integer number."

            retry_cnt += 1

        self.dialog_logger.info(Q=prompt)

        if isinstance(next_node, Invalid):
            node_history.append(next_node)  # save the last invalid
            logger.info("Max retry times reached, stop interaction now.")
        elif len(set([self._start_node] + node_history)) != len(self._graph.nodes):  # target not achieved
            logger.info("Max steps reached, stop the interaction now.")

        return node_history

    def _test_tf(self, model):
        valid_nodes = self._get_valid_nodes(self._start_node, [])
        node_history = []
        teacher_node_history = []

        optim_decov_sum = self.refresh_teacher_qa()

        # no retry when teacher forcing
        for prompt, teacher_reply in self._teacher_qa_list[:-1]:
            self.dialog_logger.info(Q=prompt)

            reply = model(prompt)

            model.force(str(teacher_reply))
            self.dialog_logger.info(A=reply, T=teacher_reply)

            next_node = self.extract_answer(reply, valid_nodes)

            node_history.append(next_node)
            teacher_node_history.append(teacher_reply)
            valid_nodes = self._get_valid_nodes(teacher_reply, [self._start_node] + teacher_node_history)

        self.dialog_logger.info(Q=self._teacher_qa_list[-1][0])

        return node_history, teacher_node_history, optim_decov_sum

    def test_one_time(self, model, teacher_forcing, instruction=None):
        self.reset()
        self.reset_model(model, instruction)

        self._graph = networkx.random_tree(self.node_num).to_undirected()
        # self._start_node = random.randint(0, self.node_num-1)
        self._start_node = 0

        logger.info("Generated random graph: nodes: {}, edges: {}"
                    .format(self._graph.nodes, self._graph.edges))

        if teacher_forcing:
            model_node_history, teacher_node_history, optim_decov_sum = self._test_tf(model)
            metric = self.calc_metric_tf(model_node_history, teacher_node_history)
        else:
            model_node_history = self._test_no_tf(model)
            metric = self.calc_metric_no_tf(model_node_history)

        full_result = {}
        full_result["metric"] = metric
        full_result["output"] = dict(
            node_history=model_node_history,
            # inv_coverage_list=covs
            teacher_node_history=teacher_node_history if teacher_forcing else None
        )
        full_result["env"] = dict(
            optim_decov_sum=optim_decov_sum if teacher_forcing else None,
            nodes=list(self._graph.nodes),
            edges=list(self._graph.edges),
            start_node=self._start_node,
            teacher_forcing=teacher_forcing,
            mcq=self.mcq,
            explain_algo=self.explain_algo,
            provide_state=self.provide_state,
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
            # min=self.min,
            # max=self.max,
            graph=self._graph,
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
