import networkx
import re

from loguru import logger

from models import BFSModel
from utils import DialogLogger, Invalid, FormatInvalid, ValueInvalid, dict_mean


def extract_int(s):
    def isint(word):
        try:
            int(word)
            return True
        except ValueError:
            return False

    return [int(word) for word in re.split("\n|,| |\t|\(|\)|\[|\]|\.|{|}", s) if isint(word)]


def extract_next_level_nodes(response):
    return extract_int(response[response.index("{"):response.index("}") + 1])


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

        # self.all_path = []`
        # self.level_to_node = {}
        # self.node_to_level = {}`

    def reset(self):
        self._teacher_qa_list = []
        self.teacher.reset("")
        # self.all_path = []
        # self.level_to_node = {}
        # self.node_to_level = {}

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
                           "4. Repeat steps 2-3 until the queue is empty." \
                        #    "The algorithm is called 'breadth-first' because it explores all the vertices at the current level before moving on to the next level. " \
                        #    "In other words, it explores the graph in a level-by-level manner, from the starting vertex to the farthest vertex.\n"
                    #   "The algorithm starts from a given vertex, and explores all its adjacent vertices at the current level before moving on to the next level. " \
                    #   "The main idea behind BFS is to explore all the nodes at a given level before moving on to the next level.\n" \

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

    def _get_prompt(self, next_node, node_history):
        '''
        Generate prompt used in exploration step

        Return: prompt (string)
        '''

        if len(set(node_history + [next_node])) == len(self._graph.nodes):
            return "Well Done. You have visited all the nodes in the graph. " \
                   "Total number of steps: {}".format(len(node_history[1:] + [next_node]))

        adj_nodes = self._get_adj_nodes(next_node)

        prompt = "Adjacent nodes: {}.".format(", ".join([str(i) for i in adj_nodes]))

        if self.provide_state:
            unvisited_adj_nodes = set(adj_nodes).difference(set(node_history))
            if len(unvisited_adj_nodes) == 0:
                prompt += " You have visited all nodes adjacent to this node."
            else:
                prompt += " You have not visited node {}." \
                          .format(", ".join([str(i) for i in unvisited_adj_nodes]))
        if self.mcq:
            valid_nodes = set(
                sum(
                    [(self._get_adj_nodes(node) + [node]) for node in node_history + [next_node]],
                    start=[]
                )
            )

            valid_nodes = {str(node) for node in valid_nodes}

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

            if next_node == self._start_node or next_node not in valid_nodes:
                return ValueInvalid(next_node)
            return next_node

        try:
            next_node = int(reply)

            if next_node == self._start_node or next_node not in valid_nodes:
                return ValueInvalid(next_node)
            return next_node
        except ValueError:
            return FormatInvalid(next_node)

    def _check_bfs(self, next_node, node_history, queue_with_levels):
        '''
        Check whether `next_node` follows BFS
        Will assume the previous steps in `node_history` already follow BFS

        Return
        - boolean: if selected interface follows bfs
        '''
        # node_history[-1] != next_node

        while len(queue_with_levels):
            curr_level = queue_with_levels[0]
            curr_level = [node for node in curr_level if node not in node_history]
            queue_with_levels[0] = curr_level

            if len(curr_level):
                break

            # current level is finished, proceed to the next level
            queue_with_levels.pop(0)

        assert len(queue_with_levels), queue_with_levels

        if next_node not in queue_with_levels[0]:
            return False, queue_with_levels

        queue_with_levels[0].pop(queue_with_levels[0].index(next_node))
        queue_with_levels.append(
            [node for node in self._get_adj_nodes(next_node) if node not in node_history]
        )
        return True, queue_with_levels

    def calc_decoverage(self, next_node, node_history):
        return 1 - len(set(node_history + [next_node])) / len(self._graph.nodes)

    def refresh_teacher_qa(self):
        self.reset_model(self.teacher, verbose=False)
        self._teacher_qa_list = []

        response = ""
        prompt = self._get_prompt(self._start_node, [])
        decov_sum = self.calc_decoverage(self._start_node, [])

        next_node = self._start_node
        node_history = [self._start_node]

        # while exist node not visited
        while len(set(self._graph.nodes).difference(set(node_history))) != 0:
            response = self.teacher(prompt)
            next_node = int(response)

            self._teacher_qa_list.append((prompt, next_node))
            prompt = self._get_prompt(next_node, node_history)

            decov_sum += self.calc_decoverage(next_node, node_history)
            node_history.append(next_node)

        self._teacher_qa_list.append((prompt, None))

        return decov_sum

    def calc_metric_no_tf(self, node_history):
        # TODO: remove the starting node in all `node_history`
        assert len(node_history) > 1

        decov_list = [self.calc_decoverage(self._start_node, [])]
        highest_cnt = 0
        check_bfs_flag = True

        queue_with_levels = [self._get_adj_nodes(node_history[0])]

        for idx, node in enumerate(node_history[1:]):  # remove the starting node
            if isinstance(node, Invalid):
                assert idx == len(node_history[1:]) - 1, \
                    f"Only the last node can be Invalid without teacher forcing. {node_history}"
                break

            # `check_bfs_flag` will remain `True` until `model` stops following bfs
            if check_bfs_flag:
                check_bfs_flag, queue_with_levels = self._check_bfs(
                    node, node_history[:idx + 1], queue_with_levels
                )
                if check_bfs_flag:
                    highest_cnt = idx + 1

            decov = self.calc_decoverage(node, node_history[:idx + 1])
            assert decov <= decov_list[-1], "`decov_list` should be a non-ascent sequence"
            decov_list.append(decov)

        acc = highest_cnt / len(node_history[1:])  # ignore the starting node
        min_decov = decov_list[-1]
        sum_decov = sum(decov_list)

        metrics = {"acc": acc, "min_decov": min_decov, "sum_decov": sum_decov}
        return metrics

    def calc_metric_tf(self, node_history, teacher_node_history):
        assert len(node_history) > 1

        decov_list = [self.calc_decoverage(self._start_node, [])]
        bfs_cnt = 0

        queue_with_levels = [self._get_adj_nodes(node_history[0])]

        for idx, (node, teacher_node) in enumerate(
            zip(node_history[1:], teacher_node_history[1:])
        ):
            if isinstance(node, Invalid):
                decov_list.append(decov_list[-1])
                continue

            check_bfs_flag, queue_with_levels = self._check_bfs(
                node, teacher_node_history[:idx + 1], queue_with_levels
            )

            if check_bfs_flag:
                bfs_cnt += 1

            queue_with_levels[0].append(node)
            queue_with_levels[0].pop(queue_with_levels[0].index(teacher_node))
            queue_with_levels[-1] = [n for n in self._get_adj_nodes(teacher_node)
                                     if n not in teacher_node_history[:idx + 1]]

            decov = self.calc_decoverage(node, teacher_node_history[:idx + 1])
            assert decov <= decov_list[-1], "`decov_list` should be a non-ascent sequence"
            decov_list.append(decov)

        acc = bfs_cnt / len(node_history[1:])  # ignore the starting node
        min_decov = decov_list[-1]
        sum_decov = sum(decov_list)

        metrics = {"acc": acc, "min_decov": min_decov, "sum_decov": sum_decov}
        return metrics

    def _test_no_tf(self, model):
        '''
        Return:
        - accuracy: percentage of node selected following bfs
        - decov_list: list of (1 - coverages)
        - trace of node explored by model
        '''
        prompt = self._get_prompt(self._start_node, [])
        node_history = [self._start_node]

        retry_cnt = 0

        value_valid_nodes = set([self._start_node] + self._get_adj_nodes(self._start_node))

        while (
            len(set(node_history)) != len(self._graph.nodes) and
            (len(node_history) - 1) < self.max_step and retry_cnt < (self.max_retry + 1)
        ):
            self.dialog_logger.info(Q=prompt)

            reply = model(prompt)
            self.dialog_logger.info(A=reply)

            # start processing response in this iteration
            next_node = self.extract_answer(reply, value_valid_nodes)

            # if `reply` is formatted, force the new reply
            if not isinstance(next_node, FormatInvalid) \
               and str(getattr(next_node, "output", next_node)) != reply:
                assert self.format_tolerant
                formatted = str(getattr(next_node, "output", next_node))
                logger.info(f"Format tolerance enabled, force the model reply to {formatted}.")
                model.force(formatted)

            if not isinstance(next_node, Invalid):
                prompt = self._get_prompt(next_node, node_history)
                node_history.append(next_node)
                retry_cnt = 0

                value_valid_nodes = value_valid_nodes.union(self._get_adj_nodes(next_node))
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
        elif len(set(node_history)) != len(self._graph.nodes):  # target not achieved
            logger.info("Max steps reached, stop the interaction now.")

        return node_history

    def _test_tf(self, model):
        value_valid_nodes = set([self._start_node] + self._get_adj_nodes(self._start_node))
        node_history = [self._start_node]
        teacher_node_history = [self._start_node]

        optim_decov_sum = self.refresh_teacher_qa()

        # no retry when teacher forcing
        for prompt, teacher_reply in self._teacher_qa_list[:-1]:
            self.dialog_logger.info(Q=prompt)

            reply = model(prompt)

            model.force(str(teacher_reply))
            self.dialog_logger.info(A=reply, T=teacher_reply)

            next_node = self.extract_answer(reply, value_valid_nodes)

            node_history.append(next_node)
            teacher_node_history.append(teacher_reply)
            value_valid_nodes = value_valid_nodes.union(self._get_adj_nodes(teacher_reply))

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
