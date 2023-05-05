import random
import networkx
from networkx import is_connected
import re
from itertools import permutations
from collections import deque
import difflib
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


def extract_int(s):
    def isint(word):
        try:
            int(word)
            return True
        except ValueError:
            return False

    return [int(word) for word in re.split("\n|,| |\t|\(|\)|\[|\]", s) if isint(word)]


def bfs_ground_truths(graph, start):
    nodes = graph.nodes
    edges = graph.edges

    adjacency_list = {node: [] for node in nodes}
    for edge in edges:
        adjacency_list[edge[0]].append(edge[1])
        adjacency_list[edge[1]].append(edge[0])

    visited = {node: False for node in nodes}
    distance = {node: float('inf') for node in nodes}
    path = {node: [] for node in nodes}
    results = []

    def bfs_visit(queue, visited, distance, path, history):
        if not queue:
            if (history, distance, path) not in results:
                results.append((history, distance, path))
            return

        current_node, visiting_seq = queue.popleft()
        history.append(current_node)

        for perm in permutations(adjacency_list[current_node]):
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

            bfs_visit(new_queue, new_visited, new_distance, new_path, new_history)

    queue = deque([(start, [])])
    visited[start] = True
    distance[start] = 0
    path[start] = [[start]]
    bfs_visit(queue, visited, distance, path, [])

    results = [list(result) for result in results]

    for result in results:
        result[2] = {item[0]: item[1][0] for item in result[2].items()}

    return results


def get_coverage(path, nodes):
    nodes = set(nodes)
    return len(nodes.intersection(set(path))) / len(nodes)


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


def extract_next_node(response):
    flag = False
    digits = []
    for x in response:
        if x.isdigit():
            digits.append(x)
            flag = True
        elif flag:
            break
    return int(''.join(digits))


class BfsEvaluator():
    def __init__(self):
        self.dialog_logger = DialogLogger(order=["Q", "A", "T"])
        self.ground_truth_model = BFSModel()

    def reset(self):
        self.dialog_logger = DialogLogger(order=["Q", "A", "T"])
        self.ground_truth_model = BFSModel()

    def init_model(self, model, teacher_forcing=False, explain_algo=False):
        prompt = "Let's play a game, one question and one answer at a time." \
                 "Now you are on a special unweighted undirected graph A. " \
                 "This undirected graph contains a series of nodes and edges, all of which have the same weight." \
                 "You can tell me which node you need to move next by saying 'move to node n (ID number)'." \
                 "For each node you move to, I will tell you the set of neighbouring nodes B of that node " \
                 "so that you can get a full picture of the graph, and you can choose to move to any known node." \
                 "Please use the BFS algorithm to solve the shortest path problem, " \
                 "which means that each move you make needs to follow the BFS algorithm\n" \


        if explain_algo:
            prompt += "BFS stands for Breadth First Search. " \
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

        # if teacher_forcing:
        #     prompt += "If multiple unexplored interfaces exist, break tie by selecting interface with smaller id value"

        prompt += "If you already know the answer, include 'over' in your answer, " \
                  "ask for as few moves as possible and give the shortest distance from the initial node to all other nodes." \
                  "At the end of the game, provide an answer in the following format: distance: {...} " \
                  "That is, answer using a dictionary format " \
                  "where the key in the dictionary is the ID number of all nodes in the corresponding graph." \
                  "During the game, your answer should be in the form: move to n,n means ID number" \
                  "If you get it, answer OK, " \
                  "then I will tell you the nodes contained in the undirected graph, the initial node IDs and the corresponding set of neighbouring nodes B." \
                  "Reply 'OK' if you understand."

        self.dialog_logger.info(Q=prompt)

        reply = model(prompt).lower()
        if teacher_forcing:
            self.dialog_logger.info(A=reply, T="ok")
            model.force("ok")
            return True

        self.dialog_logger.info(A=reply)
        return "ok" in reply

    def test_one_time(self, model, teacher_forcing=False, mcq=False, explain_algo=False, provide_state=False):

        self.reset()
        if not self.init_model(model, teacher_forcing, explain_algo):
            raise RuntimeError("failed to init model")

        # generate graph and start node
        # node_num = 4
        node_num = random.randint(3, 10)
        graph = generate_graph(node_num)
        start_node = random.randint(0, node_num - 1)

        logger.info('Generated random graph: nodes: {}, edges: {}'.format(graph.nodes, graph.edges))

        # prepare param
        curr_node = start_node
        node_history = [start_node]
        matching_cnt = 0
        adjacent_nodes = [edge[1] for edge in graph.edges(curr_node)]

        retry_cnt = 0
        response_cnt = 0
        model_response = ""

        # START
        prompt = f"START. " \
                 f"Generated graph contains nodes: {graph.nodes}. " \
                 f"The start node is {curr_node}, " \
                 f"Adjacent nodes of node{curr_node} are {adjacent_nodes}"

        if mcq:
            prompt += "Choose response from the following selections: 'over. distance is N', {}".format(', '.join(["'move to " + str(i) + "'" for i in graph.nodes]))

        while response_cnt < 40 and retry_cnt <= 3:
            self.dialog_logger.info(Q=prompt)
            # model response
            model_response = model(prompt).lower()
            # self.dialog_logger.info(A=model_response)

            if not teacher_forcing:
                self.dialog_logger.info(A=model_response)
                teacher_response = ""
            else:
                teacher_response = self.ground_truth_model(prompt=prompt)
                self.dialog_logger.info(A=model_response, T=teacher_response)
                model.force(teacher_response)

            try:
                if teacher_forcing:
                    if "distance" in teacher_response:
                        break

                # decide if model finished
                if "distance" in model_response:
                    break

                # start processing response in this iteration
                if "move" in model_response:
                    # refresh param
                    next_node = extract_next_node(model_response)

                    if provide_state:
                        unused_interfaces = set(graph.nodes).difference(node_history)
                        if len(unused_interfaces) == 0:
                            prompt += " You have visited all nodes of the graph."
                        else:
                            prompt += " You have not visited node{}.".format(
                                ', '.join([str(i) for i in unused_interfaces]))

                    if mcq:
                        prompt += "Choose response from the following selections: 'over. distance is N', {}".format(
                            ', '.join(["'move to " + str(i) + "'" for i in graph.nodes]))

                    matched = False
                    if teacher_forcing:
                        # if ground truth, record if target response is selecting same as model response
                        teacher_next_node = extract_int(teacher_response)[0]
                        matched = (teacher_next_node == next_node)

                        # l0 teacher force path always follow ground truth path
                        next_node = teacher_next_node

                    curr_node = next_node
                    adjacent_nodes = [edge[1] for edge in graph.edges(next_node)]
                    node_history.append(curr_node)

                    if matched:
                        matching_cnt += 1

                    response_cnt += 1
                    retry_cnt = 0

                    prompt = f"Adjacent nodes of node{next_node} are {adjacent_nodes}"
            except:
                if retry_cnt == 0:
                    prompt = "Invalid response. " \
                             "Try again. Please do not include any reasoning in your response. " \
                             + prompt

                retry_cnt += 1
        # TODO
        if "distance" not in model_response:
            raise RuntimeError("distance not in model_response")

        retry_cnt = 0
        while "distance" not in model_response and retry_cnt <= 3:
            prompt = "Fill the sentence by replacing the word N with a dict: \n" \
                     "over! distance: N\n" \
                     "Try again. Please do not include any reasoning in your response."
            # retry
            model_response = model(prompt).lower()
            self.dialog_logger.info(A=model_response)

            retry_cnt += 1

        if retry_cnt > 3:
            logger.warning("failed to respond valid answer within 3 retry")
            raise RuntimeError()

        # extract the dict of min distance from response
        distance = model_response[model_response.index("{"):model_response.index("}") + 1]
        logger.info(f"Distance: {distance}")
        logger.info(f"Node history: {node_history}")

        # correct bfs results, used to compare with model response
        # all_results is a list. Each element of it is also a list which is like (node_history, distance, min_dist_path)
        all_results = bfs_ground_truths(graph, start_node)

        # True or False
        correctness = difflib.SequenceMatcher(None, str(all_results[0][1]), str(distance)).ratio() == 1
        min_edit_dist = 0x7fffffff
        opt_result = []

        for result in all_results:
            cur_node_history = result[0]
            d = get_min_edit_distance(node_history, cur_node_history)
            if min_edit_dist > d:
                min_edit_dist = d
                opt_result = cur_node_history

        coverage = get_coverage(node_history, graph.nodes)

        logger.info(f'shortest correctness is {correctness}')
        logger.info(f'visit history similarity is {min_edit_dist}')
        logger.info(f"coverage: {coverage}")
        logger.info(f"accuracy: {matching_cnt / response_cnt}")

        return matching_cnt, response_cnt, correctness, min_edit_dist / len(opt_result), coverage, node_history, opt_result

    def test_multi_time_l1(self, model, times, teacher_forcing_mode, mcq, explain_algo, provide_state):
        teacher_forcing = teacher_forcing_mode == "l1"

        correctness_list = []
        min_edit_dist_list = []
        coverages = []
        matching_ratios = []
        node_histories = []
        opt_results = []
        # record fail cnt
        fail_cnt = 0

        # run multi times
        for i in range(times):
            try:
                matching_cnt, response_cnt, correctness, min_edit_dist, coverage, node_history, opt_result = \
                    self.test_one_time(model, teacher_forcing, mcq, explain_algo)

                logger.info(f"Evaluation metric #{i}: response cnt {response_cnt}, min_edit_dist {min_edit_dist}")

                correctness_list.append(correctness)
                min_edit_dist_list.append(min_edit_dist)
                coverages.append(coverage)
                matching_ratios.append(matching_cnt / response_cnt)
                node_histories.append(node_history)
                opt_results.append(opt_result)

            except Exception as e:
                logger.warning("failed this trial, error: {}".format(e))
                fail_cnt += 1

        return correctness_list, min_edit_dist_list, coverages, matching_ratios, node_histories, opt_results, fail_cnt

    def test_multi_time(self, model, times, teacher_forcing_mode="l0", mcq=False, explain_algo=False,
                        provide_state=False):
        assert teacher_forcing_mode in ["l0", "l1", "l2", "l3", "l4"], teacher_forcing_mode

        if teacher_forcing_mode in ["l0", "l1"]:
            return self.test_multi_time_l1(model, times, teacher_forcing_mode, mcq, explain_algo, provide_state)

        raise NotImplementedError(teacher_forcing_mode)
