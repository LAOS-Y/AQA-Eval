import random
import re

import networkx
from networkx import is_connected
from itertools import permutations
from collections import deque
from loguru import logger

from models import BFSModel
from utils import DialogLogger


def generate_graph(node_num):
    generation_success = False
    graph = None
    while not generation_success:
        graph = networkx.watts_strogatz_graph(node_num, 3, 0.2).to_undirected()
        generation_success = is_connected(graph)

    return graph


def bfs_ground_truths(graph, start):
    nodes = graph.nodes
    edges = graph.edges

    graph_dict = {node: [] for node in nodes}
    for edge in edges:
        graph_dict[edge[0]].append(edge[1])
        graph_dict[edge[1]].append(edge[0])

    visited = {node: False for node in nodes}
    distance = {node: float('inf') for node in nodes}
    path = {node: [] for node in nodes}
    results = []

    def bfs_all_path(queue, visited, distance, path, history):
        if not queue:
            if (history, distance, path) not in results:
                results.append((history, distance, path))
            return

        current_node, visiting_seq = queue.popleft()
        history.append(current_node)

        for perm in permutations(graph_dict[current_node]):
            new_queue = deque(queue)
            new_visited = visited.copy()
            new_distance = distance.copy()
            new_path = {node: path[node].copy() for node in nodes}
            new_history = history.copy()

            for neighbor in perm:
                new_visiting_seq = visiting_seq + [current_node]
                if not new_visited[neighbor]:
                    new_visited[neighbor] = True
                    new_distance[neighbor] = new_distance[current_node] + 1
                    new_path[neighbor] = [p + [neighbor] for p in new_path[current_node]]
                    new_queue.append((neighbor, new_visiting_seq))

            bfs_all_path(new_queue, new_visited, new_distance, new_path, new_history)

    def bfs_level(graph, start_node):
        levels = {start_node: 0}
        level_to_node = {0: [start_node]}

        queue = deque([(start_node, 0)])

        while queue:
            node, level = queue.popleft()

            for neighbor in graph[node]:
                if neighbor not in levels:
                    levels[neighbor] = level + 1
                    level_to_node[level + 1] = level_to_node.get(level + 1, []) + [neighbor]

                    queue.append((neighbor, level + 1))

        node_to_level = {node: level for level, nodes in level_to_node.items() for node in nodes}

        return level_to_node, node_to_level

    queue = deque([(start, [])])
    visited[start] = True
    distance[start] = 0
    path[start] = [[start]]
    bfs_all_path(queue, visited, distance, path, [])

    results = [list(result) for result in results]

    for result in results:
        result[2] = {item[0]: item[1][0] for item in result[2].items()}

    level_to_node, node_to_level = bfs_level(graph_dict, start)

    return results, level_to_node, node_to_level


def get_min_edit_distance(path, target):
    n = len(path)
    m = len(target)

    if n == 0 or m == 0:
        return n + m

    D = [[0] * (m + 1) for _ in range(n + 1)]

    for i in range(n + 1):
        D[i][0] = i
    for i in range(m + 1):
        D[0][i] = i

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            diag_loss = D[i - 1][j - 1]
            if not path[i - 1] == target[j - 1]:
                diag_loss += 1
            D[i][j] = min(D[i - 1][j] + 1, D[i][j - 1] + 1, diag_loss)

    return D[n][m]


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


class BfsEvaluator():
    def __init__(self, node_num=4):
        self.node_num = node_num
        self.teacher_qa_list = []
        self.dialog_logger = DialogLogger(order=["System", "Q", "A", "T"])
        self.teacher = BFSModel()
        self.all_path = []
        self.level_to_node = {}
        self.node_to_level = {}

    def reset(self):
        self.teacher_qa_list = []
        self.dialog_logger = DialogLogger(order=["System", "Q", "A", "T"])
        self.teacher = BFSModel()
        self.all_path = []
        self.level_to_node = {}
        self.node_to_level = {}

    @property
    def default_insturction(self):
        return "Let's play a game, one question and one answer at a time." \
                "Now you are on a special unweighted undirected graph A. " \
                "This undirected graph contains a series of nodes and edges, all of which have the same weight." \
                "First I will tell you the start node, and a list named B that contains all the neighbouring nodes of the start node." \
                "You can choose some nodes from B to traverse next, and then tell me nodes you choose in a set format." \
               "For example, '{0, 1, 2}' means you choose node0, node1 and node2." \
               "Please do not include any reasoning in your response, only response a pure set." \
               "The nodes that you choose from B formed list C" \
                "After you tell me C, i will tell you a new list B." \
                "This list B contains the neighbouring nodes of each node in C." \
                "And then, again you tell me the set of nodes that you will traverse next by saying a set like '{0, 1, 2}'" \
                "The cycle continues until the end of the game."\
                "The game will finish once you have visited all the nodes in the graph. \n" \
                "If you visited all the nodes in the graph, response me with 'I have visited all nodes of the graph.'" \
                "Please traverse the entire graph in as few rounds as possible." \
                "If you get it, answer OK, " \
                "then I will tell you the start node ID. " \
                "You can assume that you have traversed the start node, which means you the nodes B that you will traverse next should not contains the start node."

    def test_one_time(self, model, teacher_forcing=False, mcq=False, explain_algo=False, provide_state=False, instruction=None):
        self.reset()
        self.reset_model(model, instruction, explain_algo)

        # generate graph and start node
        self.graph = generate_graph(self.node_num)
        logger.info('Generated random graph: nodes: {}, edges: {}'.format(self.graph.nodes, self.graph.edges))
        start_node = random.randint(0, self.node_num - 1)

        # all_results is a list. Each element of it is also a list which is like (node_history, distance, min_dist_path)
        all_results, level_to_node, node_to_level = bfs_ground_truths(self.graph, start_node)
        self.all_path = [result[0] for result in all_results]
        self.level_to_node = level_to_node
        self.node_to_level = node_to_level

        if teacher_forcing:
            coverages, min_edit_distances, optim_cov_sum, model_node_history = self._test_tf(model, start_node, mcq, provide_state)
        else:
            coverages, min_edit_distances, model_node_history = self._test_no_tf(model, start_node, mcq, provide_state)

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
                nodes=list(self.graph.nodes),
                edges=list(self.graph.edges),
                start_node=start_node,
                teacher_forcing=teacher_forcing,
                mcq=mcq,
                explain_algo=explain_algo,
                provide_state=provide_state,
                instruction=self.default_insturction if instruction is None else instruction
            ),
            "history": dict(
                model_history=model.history,
                teacher_history=self.teacher_qa_list if teacher_forcing else None
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

    def _test_no_tf(self, model, start_node, mcq=False, provide_state=False):
        # prepare param
        cur_level = 0
        retry_cnt = 0
        response_cnt = 0

        # metric
        coverages = [1 - 1 / len(self.graph.nodes)]
        min_edit_distances = []

        node_history = [start_node]

        # START
        adjacent_nodes = [edge[1] for edge in self.graph.edges(start_node)]
        prompt = f"START. " \
                 f'The start node is {start_node}. ' \
                 f"Adjacent nodes of node{start_node} are {adjacent_nodes}"

        while response_cnt < 40 and retry_cnt <= 3:
            response_cnt += 1

            # response
            self.dialog_logger.info(Q=prompt)
            model_response = model(prompt).lower()
            self.dialog_logger.info(A=model_response)

            # analyse model response
            try:
                if "have visited all nodes" in model_response:
                    break

                # update level
                next_level = cur_level + 1

                cur_coverage, cur_min_edit_distance, cur_level_nodes = self.single_step_metric(model_response, next_level, node_history, teacher_forcing=False)

                min_edit_distances.append(cur_min_edit_distance)
                coverages.append(1 - cur_coverage)

                prompt = self.generate_exploring_prompt(cur_level_nodes, node_history, mcq, provide_state)

                # have traversed all nodes
                if cur_coverage == 1.0:
                    break

                retry_cnt = 0
                cur_level = next_level
            except Exception as e:
                logger.error(e)
                if retry_cnt == 0:
                    prompt = "Invalid response. " \
                             "Try again. Please do not include any reasoning in your response. " \
                             + prompt
                retry_cnt += 1
        return coverages, min_edit_distances, node_history

    def _test_tf(self, model, start_node, mcq, provide_state):
        node_history = [start_node]
        cur_level = 0

        coverages = [1 - 1 / len(self.graph.nodes)]
        min_edit_distances = []

        optim_cov_sum = self.refresh_teacher_qa(start_node, mcq, provide_state)

        # no retry when teacher forcing
        for prompt, teacher_reply in self.teacher_qa_list:
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
        self.teacher_qa_list = []

        # prepare param
        cur_level = 0
        node_history = [start_node]

        # START
        adjacent_nodes = [edge[1] for edge in self.graph.edges(start_node)]
        prompt = f"START. " \
                 f'The start node is {start_node}. ' \
                 f"Adjacent nodes of node{start_node} are {adjacent_nodes}"

        cov_sum = 0
        while len(set(self.graph.nodes).difference(set(node_history))) != 0:  # while exist node not visited
            response = self.teacher(prompt)
            self.teacher_qa_list.append((prompt, response))
            cur_level += 1
            cur_level_nodes_ground_truth = self.level_to_node.get(cur_level)
            node_history.extend(cur_level_nodes_ground_truth)
            prompt = self.generate_exploring_prompt(cur_level_nodes_ground_truth, set(node_history), mcq, provide_state)
            cov_sum += len(set(node_history)) / len(self.graph.nodes)

        return cov_sum

    def generate_exploring_prompt(self, cur_level_nodes, node_history, mcq, provide_state):
        adjacent_nodes = list(
            set([edge[1] for next_node in cur_level_nodes for edge in self.graph.edges(next_node)]))

        prompt = f"Adjacent nodes of nodes{cur_level_nodes} are {adjacent_nodes}."

        if provide_state:
            no_visited_nodes = set(self.graph.nodes) - set(node_history)
            if len(no_visited_nodes) == 0:
                prompt += " You have visited all nodes of the graph."
            else:
                prompt += " You have not visited node {}.".format(
                    ', '.join([str(i) for i in no_visited_nodes]))

        return prompt

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
        cur_coverage = len(set(node_history)) / len(self.graph.nodes)
        return cur_coverage, cur_min_edit_distance, cur_level_nodes
