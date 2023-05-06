import networkx
import random
import re

from loguru import logger

from models.dfs_model import DFSModel
from utils import DialogLogger, Invalid, FormatInvalid, ValueInvalid, dict_mean


def extract_int(s):
    def isint(word):
        try:
            int(word)
            return True
        except ValueError:
            return False
    return [int(word) for word in re.split("\n|,|\.| |\'", s) if isint(word)]


def get_coverage(path, nodes):
    nodes = set(nodes)
    return len(nodes.intersection(set(path))) / len(nodes)


class DFSEvaluator():
    def __init__(
        self, node_num=10, explain_algo=True, mcq=True, provide_state=True,
        format_tolerant=True
    ):
        self.node_num = node_num
        self.explain_algo = explain_algo
        self.mcq = mcq
        self.provide_state = provide_state
        self.format_tolerant = format_tolerant
        self.teacher = DFSModel()
        self.dialog_logger = DialogLogger(order=["System", "Q", "A", "T"])

    def reset(self):
        self.teacher.reset("")
        self._teacher_qa_list = None
        self._graph = None
        self._start_node = None

    @property
    def default_insturction(self):
        instruction = "You are required to visit all the nodes in an undirected non-cyclic graph." \
                      "An undirected non-cyclic garph contains a set of node, and a set of edges that each connects a pair of nodes. " \
                      "All edges are undirected, so that you can move from one node to the other connected by the edge in either direction. " \
                      "Every time you visit a node, you will be given the adjacent nodes connected to this node. " \
                      "You can only reply with a integer number indicating which node to be visited next. " \
                      "Try moving as few times as you can. " \
                      "You are currently on the node 0." \

        if self.explain_algo:
            instruction += "\nYou should use depth first search algorithm, each time you should " \
                           "select a node you have not moved to. If all nodes adjacent to the " \
                           "current node have been visited, you should back track to the node " \
                           "through which you entered this node for the first time. "

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

    def _get_prompt(self, curr_node, node_history):
        '''
        Generate prompt used in exploration step

        Return: prompt (string)
        '''
        # TODO: rename `unused_nodes` to `unvisited_adj_nodes`
        adj_nodes = self._get_adj_nodes(curr_node)

        prompt = "Adjacent nodes: {}.".format(", ".join([str(i) for i in adj_nodes]))

        if self.provide_state:
            unused_nodes = set(adj_nodes).difference(set(node_history))
            if len(unused_nodes) == 0:
                prompt += " You have visited all nodes adjacent to this node."
            else:
                prompt += " You have not visited node {}." \
                          .format(", ".join([str(i) for i in unused_nodes]))
        if self.mcq:
            prompt += " Choose the next node to visit: {}." \
                      .format(", ".join([str(i) for i in adj_nodes]))

        return prompt

    def extract_answer(self, reply, adj_nodes):
        # parse reply from model and return the formated answer
        # return an `Invalid` if failed to do so
        if self.format_tolerant:
            nums = re.findall(r'\d+', reply)
            if not len(nums):
                return FormatInvalid(reply)

            next_node = int(nums[0])

            if next_node not in adj_nodes:
                return ValueInvalid(next_node)
            return next_node

        try:
            next_node = int(reply)

            if next_node not in adj_nodes:
                return ValueInvalid(next_node)
            return next_node
        except ValueError:
            return FormatInvalid(next_node)

    def _check_dfs(self, next_node, node_history):
        '''
        Check whether `next_node` follows DFS
        Will assume the previous steps in `node_history` already follow DFS

        Return
        - boolean: if selected interface follows dfs
        '''
        curr_node = node_history[-1]
        adj_nodes = self._get_adj_nodes(curr_node)

        # check if model selected node following dfs path
        # i.e. select unvisited child node or parent node
        unvisited_adj_nodes = set(adj_nodes).difference(set(node_history))
        if len(unvisited_adj_nodes):
            # should visit child node
            return next_node in unvisited_adj_nodes

        # if all child have been fisited,
        # check if model is visiting its parent node in the history stack

        curr_node_idx = node_history.index(curr_node)
        # `curr_node` should be the root only when there are children of the root unvisited
        assert curr_node_idx
        # should visit father node
        if_dfs = (next_node == node_history[curr_node_idx - 1])

        return if_dfs

    def calc_decoverage(self, next_node, node_history):
        return 1 - len(set(node_history + [next_node])) / len(self._graph.nodes)

    def refresh_teacher_qa(self):
        self.reset_model(self.teacher, verbose=False)
        self._teacher_qa_list = []

        response = ""
        prompt = self._get_prompt(self._start_node, [])
        decov_sum = self.calc_decoverage(self._start_node, [])

        curr_node = self._start_node
        node_history = [self._start_node]

        # while exist node not visited
        while len(set(self._graph.nodes).difference(set(node_history))) != 0:
            response = self.teacher(prompt)
            self._teacher_qa_list.append((prompt, response))

            curr_node = extract_int(response)[0]
            prompt = self._get_prompt(curr_node, node_history)
            decov_sum += self.calc_decoverage(curr_node, node_history)
            node_history.append(curr_node)

        return decov_sum

    def _test_no_tf(self, model):
        '''
        Return:
        - accuracy: percentage of node selected following dfs
        - decov_list: list of (1 - coverages)
        - trace of node explored by model
        '''
        correct_cnt = 0
        decov_list = [self.calc_decoverage(self._start_node, [])]
        dfs_flag = True

        curr_node = self._start_node
        node_history = [self._start_node]
        prompt = self._get_prompt(self._start_node, [])

        cnt = 0
        retry_cnt = 0

        while cnt < 20 and retry_cnt <= 3:
            self.dialog_logger.info(Q=prompt)

            reply = model(prompt).lower()
            self.dialog_logger.info(A=reply)

            # start processing response in this iteration
            next_node = self.extract_answer(reply, self._get_adj_nodes(curr_node))
            if isinstance(next_node, Invalid):
                # if `reply` is formatted, force the new reply
                if self.format_tolerant and isinstance(next_node, ValueInvalid):
                    formatted = next_node.output
                    assert isinstance(formatted, int)
                    logger.info(f"Format tolerance enabled, force the model reply to {formatted}.")
                    model.force(str(formatted))

                if retry_cnt == 0:
                    prompt = "Invalid response. Try again. Please do not include any reasoning " \
                             "in your response. " + prompt
                retry_cnt += 1
                continue

            # `dfs_flag` will remain `True` until `model` stops following dfs
            dfs_flag = dfs_flag and self._check_dfs(next_node, node_history)
            if dfs_flag:
                correct_cnt += 1

            decov = self.calc_decoverage(next_node, node_history)
            decov_list.append(decov)

            cnt += 1
            retry_cnt = 0

            # if all node visited, finish
            if decov == 0.0:
                break

            node_history.append(next_node)
            curr_node = next_node
            prompt = self._get_prompt(curr_node, node_history)

        if cnt == 0:
            return 0, decov_list, node_history
        return correct_cnt / cnt, decov_list, node_history

    def _test_tf(self, model):
        '''
        Return:
        - accuracy: percentage of node selected following dfs
        - decov_list: list of (1 - coverages)
        - trace of node explored by model
        '''
        correct_cnt = 0
        decov_list = [self.calc_decoverage(self._start_node, [])]

        curr_node = self._start_node
        node_history = [self._start_node]

        optim_decov_sum = self.refresh_teacher_qa()

        # no retry when teacher forcing
        for prompt, teacher_reply in self._teacher_qa_list:
            self.dialog_logger.info(Q=prompt)

            reply = model(prompt)

            model.force(teacher_reply)
            self.dialog_logger.info(A=reply, T=teacher_reply)

            next_node = self.extract_answer(reply, self._get_adj_nodes(curr_node))

            if isinstance(next_node, Invalid):
                # coverage value does not change
                dfs_correct = False
                decov = decov_list[-1]
            else:
                dfs_correct = self._check_dfs(next_node, node_history)
                decov = self.calc_decoverage(next_node, node_history)

            if dfs_correct:
                correct_cnt += 1
            decov_list.append(decov)
            print(decov)

            curr_node = int(teacher_reply)
            node_history.append(curr_node)

        return correct_cnt / len(self._teacher_qa_list), decov_list, optim_decov_sum, node_history

    def test_one_time(self, model, teacher_forcing, instruction=None):
        self.reset()
        self.reset_model(model, instruction)

        self._graph = networkx.random_tree(self.node_num).to_undirected()
        # self._start_node = random.randint(0, self.node_num-1)
        self._start_node = 0

        logger.info("Generated random graph: nodes: {}, edges: {}"
                    .format(self._graph.nodes, self._graph.edges))

        if teacher_forcing:
            accuracy, covs, optim_decov_sum, model_node_history = self._test_tf(model)
        else:
            accuracy, covs, model_node_history = self._test_no_tf(model)

        metric = {"accuracy": accuracy, "min_decov": covs[-1], "sum_decov": sum(covs)}

        full_result = {}
        full_result["metric"] = metric
        full_result["output"] = dict(
            guess_list=model_node_history,
            # inv_coverage_list=covs
            # TODO: return `teacher_guess_list`
            teacher_guess_list=None,
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
