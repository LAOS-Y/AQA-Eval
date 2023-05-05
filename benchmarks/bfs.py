import random
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
        # 存储每个节点所在的层数和遍历结果
        levels = {start_node: 0}
        level_to_node = {0: [start_node]}

        # 初始化队列
        queue = deque([(start_node, 0)])

        # 遍历队列
        while queue:
            # 取出队列中的第一个元素
            node, level = queue.popleft()

            # 遍历该节点的所有邻居节点
            for neighbor in graph[node]:
                # 如果邻居节点未被遍历过
                if neighbor not in levels:
                    # 记录邻居节点所在的层数和遍历结果
                    levels[neighbor] = level + 1
                    level_to_node[level + 1] = level_to_node.get(level + 1, []) + [neighbor]

                    # 将邻居节点加入队列
                    queue.append((neighbor, level + 1))

        # 构造节点到层数的映射
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


def extract_next_level_nodes(response):
    return response[response.index("{"):response.index("}") + 1]


class BfsEvaluator():
    def __init__(self):
        self.dialog_logger = DialogLogger(order=["Q", "A", "T"])
        self.ground_truth_model = BFSModel()
        self.all_path = []
        self.level_to_node = []
        self.node_to_level = []

    def reset(self):
        self.dialog_logger = DialogLogger(order=["Q", "A", "T"])
        self.ground_truth_model = BFSModel()
        self.all_path = []
        self.level_to_node = []
        self.node_to_level = []

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

        # correct bfs results, used to compare with model response
        # all_results is a list. Each element of it is also a list which is like (node_history, distance, min_dist_path)
        all_results, level_to_node, node_to_level = bfs_ground_truths(graph, start_node)
        self.all_path = [result[0] for result in all_results]
        self.level_to_node = level_to_node
        self.node_to_level = node_to_level

        # prepare param
        cur_level = 0
        visited_nodes = {start_node}

        coverages = []
        min_edit_distances = []

        retry_cnt = 0
        response_cnt = 0
        model_response = ""

        # START
        prompt = f"START. " \
                 f'Generated graph: nodes: {graph.nodes}, edges: {graph.edges}. ‘ \
                 f"The start node is {start_node}. '

        # if mcq:
        #     prompt += f"Choose response from the following selections: 'over', move to {cur_node}"

        while response_cnt < 40 and retry_cnt <= 3:
            self.dialog_logger.info(Q=prompt)
            # model response
            model_response = model(prompt).lower()

            if teacher_forcing:
                teacher_response = self.ground_truth_model(prompt=prompt)
                self.dialog_logger.info(A=model_response, T=teacher_response)
                model.force(teacher_response)
            else:
                teacher_response = ""
                self.dialog_logger.info(A=model_response)

            try:
                if teacher_forcing and "over" in teacher_response:
                    break

                # decide if model finished
                if "over" in model_response:
                    break

                # start processing response in this iteration
                if "next level nodes are" in model_response:
                    # refresh param
                    next_level_nodes = extract_next_level_nodes(model_response)

                    # TODO
                    # if provide_state:
                    #     no_visited_nodes = set(graph.nodes).difference(node_history)
                    #     if len(no_visited_nodes) == 0:
                    #         prompt += " You have visited all nodes of the graph."
                    #     else:
                    #         prompt += " You have not visited node{}.".format(
                    #             ', '.join([str(i) for i in no_visited_nodes]))

                    # if mcq:
                    #     prompt += "Choose response from the following selections: 'over. distance is N', {}".format(
                    #         ', '.join(["'move to " + str(i) + "'" for i in graph.nodes]))

                    # update level info
                    cur_level += 1
                    cur_level_nodes_ground_truth = self.level_to_node.get(cur_level)

                    # record metric min_edit_distance
                    levels = [self.node_to_level.get(next_level_node) for next_level_node in next_level_nodes]
                    levels_ground_truth = [cur_level for i in cur_level_nodes_ground_truth]
                    cur_min_edit_distance = get_min_edit_distance(levels, levels_ground_truth)
                    min_edit_distances.append(cur_min_edit_distance)

                    if teacher_forcing:
                        # l0 teacher force path always follow ground truth path
                        cur_level_nodes = cur_level_nodes_ground_truth
                    else:
                        cur_level_nodes = next_level_nodes

                    # record metric coverage
                    visited_nodes.intersection(set(cur_level_nodes))
                    cur_coverage = len(visited_nodes) / len(graph.nodes)
                    coverages.append(cur_coverage)

                    response_cnt += 1
                    retry_cnt = 0

                    prompt = "If you have visited all nodes, response 'over'." \
                             "Otherwise, please continue traverse the graph in bfs algorithm and response next level nodes"
            except:
                if retry_cnt == 0:
                    prompt = "Invalid response. " \
                             "Try again. Please do not include any reasoning in your response. " \
                             + prompt

                retry_cnt += 1

        if "over" not in model_response:
            raise RuntimeError("over not in model_response")

        cov_result = 0
        for coverage in coverages:
            cov_result = cov_result + 1 - coverage

        med_result = 0
        for min_edit_distance in min_edit_distances:
            med_result = med_result + min_edit_distance

        logger.info(f"cov_result: {cov_result}")
        logger.info(f"med_result: {med_result}")

        return cov_result, med_result

    def test_multi_time_l1(self, model, times, teacher_forcing_mode, mcq, explain_algo, provide_state):
        teacher_forcing = teacher_forcing_mode == "l1"

        coverages = []
        meds = []
        fail_cnt = 0

        # run multi times
        for i in range(times):
            try:
                cov_result, med_result = self.test_one_time(model, teacher_forcing, mcq, explain_algo)

                logger.info(f"Evaluation metric #{i}: cov_result: {cov_result}, med_result: {med_result}")

                coverages.append(cov_result)
                meds.append(med_result)

            except Exception as e:
                logger.warning("failed this trial, error: {}".format(e))
                fail_cnt += 1

        return coverages, meds, fail_cnt

    def test_multi_time(self, model, times, teacher_forcing_mode="l0", mcq=False, explain_algo=False,
                        provide_state=False):
        assert teacher_forcing_mode in ["l0", "l1", "l2", "l3", "l4"], teacher_forcing_mode

        if teacher_forcing_mode in ["l0", "l1"]:
            return self.test_multi_time_l1(model, times, teacher_forcing_mode, mcq, explain_algo, provide_state)

        raise NotImplementedError(teacher_forcing_mode)
