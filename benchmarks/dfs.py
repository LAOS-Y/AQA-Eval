import networkx
import random
import re

from loguru import logger

from models.dfs_model import DFSModel
from utils import DialogLogger, dict_mean


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
    def __init__(self, node_num=10, explain_algo=True, mcq=True, provide_state=True) -> None:
        self.node_num = node_num
        self.mcq = mcq
        self.explain_algo = explain_algo
        self.provide_state = provide_state
        self.teacher = DFSModel()
        self.dialog_logger = DialogLogger(order=["System", "Q", "A", "T"])

    def reset(self):
        self.teacher.reset("")
        self._teacher_qa_list = None
        self._graph = None
        self._start_node = None

    @property
    def default_insturction(self):
        instruction = "You are on a node of an undirected non-cyclic graph. " \
                      "An undirected non-cyclic garph contains a set of node, and a set of edges that each connects a pair of nodes. " \
                      "All edges are undirected, so that you can move from one node to the other connected by the edge in either direction. " \
                      "You are asked to visit all nodes in the graph. " \
                      "Every time you enter a node, you will be given the adjacent nodes connected to this node. " \
                      "You can move to a node by responding the node ID adjacent to the node. " \
                      "Try move as few times as you can. " \
                      "The game will finish once you have visited all the nodes in the graph."

        if self.explain_algo:
            instruction += "You should use depth first search algorithm, each time you should " \
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

        adj_nodes = self._get_adj_nodes(curr_node)

        prompt = "You are on node {}, number of adjacent node is {}, adjacent nodes are {}" \
                 .format(str(curr_node), len(adj_nodes), ', '.join([str(i) for i in adj_nodes]))

        if self.provide_state:
            unused_nodes = set(adj_nodes).difference(set(node_history))
            if len(unused_nodes) == 0:
                prompt += " You have visited all nodes adjacent to this node."
            else:
                prompt += " You have not visited node {} adjacent to this node." \
                          .format(', '.join([str(i) for i in unused_nodes]))
        if self.mcq:
            prompt += " Select the next node ID from the following selections: {}" \
                      .format(', '.join([str(i) for i in adj_nodes]))

        return prompt

    def single_step_metric(self, curr_node, model_response, node_history):
        '''
        Process model response, raise error if fail to process invalid response
        AND Calculate metric values
        - model_response: model string response
        - node_history: variables for tracking node states,
                        will be appended with the selected next node

        Return
        - int: next node selected in model response
        - boolean: if selected interface follows dfs
        - float: coverage of nodes after exploration
        '''

        ints = extract_int(model_response)
        if len(ints) != 1:
            raise RuntimeError("Returning int number not equal to 1")

        next_node = ints[0]
        adj_nodes = self._get_adj_nodes(curr_node)
        if next_node not in adj_nodes:
            raise RuntimeError("Selected next node not in adjacent node")

        # check if model selected node following dfs path
        # i.e. select unvisited child node or parent node
        unused_nodes = set(adj_nodes).difference(set(node_history))
        if len(unused_nodes) == 0:
            # if all child have been fisited,
            # check if model is visiting its parent node in the history stack
            parent_node_index = node_history.index(curr_node) - 1
            if parent_node_index < 0:
                # if curr node is root, fail this correctness check
                dfs_correctness = False
            else:
                dfs_correctness = (next_node == node_history[parent_node_index])
        else:
            # should visit child node
            dfs_correctness = (next_node in unused_nodes)

        node_history.append(next_node)

        cov = len(set(node_history)) / len(self._graph.nodes)

        return next_node, dfs_correctness, cov

    def refresh_teacher_qa(self):
        self.reset_model(self.teacher, verbose=False)
        self._teacher_qa_list = []

        curr_node = self._start_node
        response = ""
        prompt = "START. " + self._get_prompt(self._start_node, [])
        node_history = [self._start_node]

        cov_sum = 0
        # while exist node not visited
        while len(set(self._graph.nodes).difference(set(node_history))) != 0:
            response = self.teacher(prompt)
            self._teacher_qa_list.append((prompt, response))

            curr_node = extract_int(response)[0]
            prompt = self._get_prompt(curr_node, node_history)
            node_history.append(curr_node)
            cov_sum += len(set(node_history)) / len(self._graph.nodes)

        return cov_sum

    def _test_no_tf(self, model):
        '''
        Return:
        - accuracy: percentage of node selected following dfs
        - covs: list of (1 - coverages)
        - trace of node explored by model
        '''

        # info required for recording and iterative eval
        cnt = 0
        curr_node = self._start_node

        prompt = "START. " + self._get_prompt(curr_node, [])

        retry_cnt = 0

        node_history = [curr_node]

        correct_cnt = 0
        covs = [1 - 1 / len(self._graph.nodes)]

        while cnt < 20 and retry_cnt <= 3:
            self.dialog_logger.info(Q=prompt)

            reply = model(prompt).lower()
            self.dialog_logger.info(A=reply)

            # start processing response in this iteration
            try:
                curr_node, dfs_correct, cov = self.single_step_metric(
                    curr_node, reply, node_history
                )

                prompt = self._get_prompt(curr_node, node_history)

                if dfs_correct:
                    correct_cnt += 1
                covs.append(1 - cov)

                cnt += 1
                retry_cnt = 0

                # if all node visited, finish
                if cov == 1.0:
                    break
            except Exception as e:
                print(e)
                if retry_cnt == 0:
                    prompt = "Invalid response. Try again. Please do not include any reasoning " \
                             "in your response. " + prompt
                retry_cnt += 1

        if cnt == 0:
            return 0, covs, node_history
        return correct_cnt / cnt, covs, node_history

    def _test_tf(self, model):
        '''
        Return:
        - accuracy: percentage of node selected following dfs
        - covs: list of (1 - coverages)
        - trace of node explored by model
        '''
        # info required for recording and iterative eval
        node_history = [self._start_node]

        correct_cnt = 0
        cov = 1 / len(self._graph.nodes)
        covs = [cov]

        optim_cov_sum = self.refresh_teacher_qa()

        # no retry when teacher forcing
        for prompt, teacher_reply in self._teacher_qa_list:
            self.dialog_logger.info(Q=prompt)

            reply = model(prompt)
            model.force(teacher_reply)
            self.dialog_logger.info(A=reply, T=teacher_reply)

            curr_node = extract_int(prompt)[0]

            try:
                _, dfs_correct, cov = self.single_step_metric(curr_node, reply, node_history)
            except Exception as e:
                print(e)
                # coverage value does not change
                dfs_correct = False

            if dfs_correct:
                correct_cnt += 1
            covs.append(1 - cov)

        return correct_cnt / len(self._teacher_qa_list), covs, optim_cov_sum, node_history

    def test_one_time(self, model, teacher_forcing, instruction=None):
        self.reset()
        self.reset_model(model, instruction)

        self._graph = networkx.random_tree(self.node_num).to_undirected()
        self._start_node = random.randint(0, self.node_num-1)

        logger.info("Generated random graph: nodes: {}, edges: {}"
                    .format(self._graph.nodes, self._graph.edges))

        if teacher_forcing:
            accuracy, covs, optim_cov_sum, model_node_history = self._test_tf(model)
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
            optim_cov_sum=optim_cov_sum if teacher_forcing else None,
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
