import random
import re

import networkx
from networkx import is_connected
from itertools import permutations
from collections import deque
from loguru import logger

from models import BFSModel
from utils import DialogLogger, Invalid, FormatInvalid, ValueInvalid, dict_mean


def generate_graph(node_num):
    graph = networkx.random_tree(node_num).to_undirected()

    return graph


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
        return "You are required to visit all the nodes in a special unweighted undirected non-cyclic graph." \
               "A unweighted undirected non-cyclic graph contains a series of nodes and edges, all of which have the same weight." \
               "First I will tell you the start node, and a list named B that contains all the neighbouring nodes of the start node." \
               "You can choose some nodes from B to traverse next, and then tell me nodes you choose in a set format." \
               "For example, '{0, 1, 2}' means you choose node0, node1 and node2." \
               "Please do not include any reasoning in your response, only response a pure set." \
               "Every time you tell me a set, you will be given the neighbouring nodes of each node in the previous set." \
               "The game will finish once you have visited all the nodes in the graph. \n" \
               "Please traverse the entire graph in as few rounds as possible."

    def test_one_time(self, model, teacher_forcing=False, instruction=None):
        self.reset()
        self.reset_model(model, instruction, self.explain_algo)

        # generate graph and start node
        self._graph = generate_graph(self.node_num)
        logger.info('Generated random graph: nodes: {}, edges: {}'.format(self._graph.nodes, self._graph.edges))
        self._start_node = 0

        if teacher_forcing:
            coverages, min_edit_distances, optim_cov_sum, model_node_history = self._test_tf(model)
        else:
            coverages, min_edit_distances, model_node_history = self._test_no_tf(model)

        # calc final metric
        cov_result = sum(coverages)
        med_result = sum(min_edit_distances)

        logger.info(f"cov_result: {cov_result}")
        logger.info(f"med_result: {med_result}")

        full_result = {
            "med_sum": med_result,
            "inv_cov_sum": cov_result,
            "inv_cov_min": coverages[-1],
            "output": dict(
                guess_list=model_node_history,
                inv_coverage_list=coverages,
                min_edit_distance_list=min_edit_distances
            ),
            "env": dict(
                optim_cov_sum=optim_cov_sum if teacher_forcing else None,
                nodes=list(self._graph.nodes),
                edges=list(self._graph.edges),
                start_node=self._start_node,
                teacher_forcing=teacher_forcing,
                mcq=self.mcq,
                explain_algo=self.explain_algo,
                provide_state=self.provide_state,
                instruction=self.default_insturction if instruction is None else instruction
            ),
            "history": dict(
                model_history=model.history,
                teacher_history=self._teacher_qa_list if teacher_forcing else None
            )
        }

        return cov_result, coverages[-1], med_result, full_result

    def reset_model(self, model, instruction=None, explain_algo=False, verbose=True):
        # clear dialog history and give instruction
        # will use `self.default_insturction` if `instruction` is None
        if instruction is None:
            instruction = self.default_insturction

        if explain_algo:
            instruction += "Please use the BFS algorithm traverse this graph.\n" \
                      "BFS stands for Breadth First Search. " \
                      "It is a popular algorithm used to traverse a graph or a tree data structure. " \
                      "The algorithm starts from a given vertex, and explores all its adjacent vertices at the current level before moving on to the next level. " \
                      "The main idea behind BFS is to explore all the nodes at a given level before moving on to the next level.\n" \
                      "The algorithm works as follows:\n" \
                      "1. Initialize a queue data structure and add the starting vertex to the queue.\n" \
                      "2. While the queue is not empty, remove the first vertex from the queue and mark it as visited.\n" \
                      "3. For each of the unvisited adjacent vertices of the removed vertex, add them to the queue.\n" \
                      "4. Repeat steps 2-3 until the queue is empty.\n" \
                      "The algorithm is called 'breadth-first' because it explores all the vertices at the current level before moving on to the next level. " \
                      "In other words, it explores the graph in a level-by-level manner, from the starting vertex to the farthest vertex.\n"
        if verbose:
            self.dialog_logger.info(System=instruction)

        model.reset(instruction)
        return

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

    def generate_exploring_prompt(self, cur_level_nodes, node_history, mcq, provide_state):
        adjacent_nodes = list(
            set([edge[1] for next_node in cur_level_nodes for edge in self._graph.edges(next_node)]))

        prompt = f"Adjacent nodes of nodes{cur_level_nodes} are {adjacent_nodes}."

        if provide_state:
            no_visited_nodes = set(self._graph.nodes) - set(node_history)
            if len(no_visited_nodes) == 0:
                prompt += " You have visited all nodes of the graph."
            else:
                prompt += " You have not visited node {}.".format(
                    ', '.join([str(i) for i in no_visited_nodes]))

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

    def _test_no_tf(self, model):
        '''
        Return:
        - accuracy: percentage of node selected following dfs
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
        node_history = [self._start_node]
        cur_level = 0

        coverages = [1 - 1 / len(self._graph.nodes)]
        min_edit_distances = []

        optim_cov_sum = self.refresh_teacher_qa()

        # no retry when teacher forcing
        for prompt, teacher_reply in self._teacher_qa_list:
            self.dialog_logger.info(Q=prompt)
            model_response = model(prompt)
            model.force(teacher_reply)
            self.dialog_logger.info(A=model_response, T=teacher_reply)

            next_level = cur_level + 1
            cur_coverage, cur_min_edit_distance, _ = self.single_step_metric(model_response, next_level, node_history, teacher_forcing=True)
            min_edit_distances.append(cur_min_edit_distance)
            coverages.append(1 - cur_coverage)
            cur_level = next_level

        return coverages, min_edit_distances, optim_cov_sum, node_history

    def refresh_teacher_qa(self, start_node, mcq, provide_state):
        self.reset_model(self.teacher, verbose=False)
        self._teacher_qa_list = []

        # prepare param
        cur_level = 0
        node_history = [start_node]

        # START
        adjacent_nodes = [edge[1] for edge in self._graph.edges(start_node)]
        prompt = f"START. " \
                 f'The start node is {start_node}. ' \
                 f"Adjacent nodes of node{start_node} are {adjacent_nodes}"

        cov_sum = 0
        while len(set(self._graph.nodes).difference(set(node_history))) != 0:  # while exist node not visited
            response = self.teacher(prompt)
            self._teacher_qa_list.append((prompt, response))
            cur_level += 1
            cur_level_nodes_ground_truth = self.level_to_node.get(cur_level)
            node_history.extend(cur_level_nodes_ground_truth)
            prompt = self.generate_exploring_prompt(cur_level_nodes_ground_truth, set(node_history), mcq, provide_state)
            cov_sum += len(set(node_history)) / len(self._graph.nodes)

        return cov_sum

    def single_step_metric(self, model_response, cur_level, node_history, teacher_forcing):
        # extract next nodes from model response
        next_level_nodes = extract_next_level_nodes(model_response)
        cur_level_nodes_ground_truth = self.level_to_node.get(cur_level)
        if cur_level_nodes_ground_truth:
            levels_ground_truth = [cur_level for _ in cur_level_nodes_ground_truth]
        else:
            levels_ground_truth = []

        # record metric min_edit_distance
        levels = [self.node_to_level.get(next_level_node) for next_level_node in next_level_nodes]
        cur_min_edit_distance = get_min_edit_distance(levels, levels_ground_truth)

        if teacher_forcing:
            cur_level_nodes = cur_level_nodes_ground_truth
        else:
            cur_level_nodes = next_level_nodes

        node_history.extend(cur_level_nodes)
        # record metric coverage
        cur_coverage = len(set(node_history)) / len(self._graph.nodes)
        return cur_coverage, cur_min_edit_distance, cur_level_nodes
