from copy import deepcopy

from models import BFSModel
from utils import Invalid
from .traverse import TraverseGraphEvaluator


class BFSEvaluator(TraverseGraphEvaluator):
    def __init__(
        self, node_num=4, explain_algo=True, mcq=False, provide_state=False,
        format_tolerant=True, max_retry=0, max_step=None, use_scene_instruction=False
    ):
        super(BFSEvaluator, self).__init__(
            node_num, explain_algo, mcq, provide_state,
            format_tolerant, max_retry, max_step, use_scene_instruction
        )
        self.teacher = BFSModel()

    @property
    def default_instruction(self):
        if not self.use_scene_instruction:
            instruction = "You are required to visit all the nodes in an undirected non-cyclic graph." \
                          "An undirected non-cyclic garph contains a set of node, and a set of edges that each connects a pair of nodes. " \
                          "Every time you visit a node, you will be given the adjacent nodes connected to this node. " \
                          "You can only visit nodes that are adjacent to the already visited nodes. " \
                          "You can only reply with a integer number indicating which node to be visited next. " \
                          "Please traverse the entire graph in as few rounds as possible." \
                          "Initially, you have already visited node 0."
        else:
            instruction = "You are the manager of a chain of restaurants. " \
                          "The current goal of the restaurant is to open a branch in every city in Country X. " \
                          "Currently, the restaurant has only one branch in City 0 of Country X. " \
                          "Considering the opening and management costs, each time you open a new branch, " \
                          "you can only choose one city from all the neighboring cities of " \
                          "the cities where you've already opened branches." \
                          "Once you choose a city, you need to reply with the city's number, " \
                          "and then you will be given a list of all the neighboring cities of that chosen city." \
                          "Please open a branch in every city in Country X as quickly as possible."

        if self.explain_algo:
            instruction += "You should use breadth first search algorithm. " \
                           "The algorithm works as follows:\n" \
                           "1. Initialize a queue data structure and add the starting node to the queue.\n" \
                           "2. While the queue is not empty, visit the first node and remove it from the queue.\n" \
                           "3. For nodes adjacent to the removed vertex, add the unvisited ones to the queue.\n" \
                           "4. Repeat steps 2-3 until the queue is empty."

        return instruction

    def _get_prompt(self, next_node, visited_nodes):
        '''
        Generate prompt used in exploration step

        Return: prompt (string)
        '''

        if len(set(visited_nodes + [next_node])) == len(self._graph.nodes):
            if not self.use_scene_instruction:
                return "Well Done. You have visited all the nodes in the graph. " \
                       "Total number of steps: {}".format(len(visited_nodes[1:] + [next_node]))

            return "Well Done. You have open branches in all cities of Country X. " \
                    "Total number of steps: {}".format(len(visited_nodes[1:] + [next_node]))

        adj_nodes = self._get_adj_nodes(next_node)

        if not self.use_scene_instruction:
            prompt = "Adjacent nodes: {}.".format(", ".join([str(i) for i in adj_nodes]))
        else:
            prompt = "Adjacent cities: {}.".format(", ".join([str(i) for i in adj_nodes]))

        if self.provide_state:
            unvisited_adj_nodes = set(adj_nodes).difference(set(visited_nodes))
            if not self.use_scene_instruction:
                if len(unvisited_adj_nodes) == 0:
                    prompt += " You have visited all nodes adjacent to this node."
                else:
                    prompt += " You have not visited node {}." \
                              .format(", ".join([str(i) for i in unvisited_adj_nodes]))
            else:
                if len(unvisited_adj_nodes) == 0:
                    prompt += " You have open branches in all cities adjacent to this city."
                else:
                    prompt += " You have not open branches in cities {}." \
                              .format(", ".join([str(i) for i in unvisited_adj_nodes]))
        if self.mcq:
            valid_nodes = self._get_valid_nodes(next_node, visited_nodes)

            valid_nodes = {str(node) for node in valid_nodes}

            if not self.use_scene_instruction:
                prompt += " Choose the next node to visit: {}.".format(", ".join(valid_nodes))
            else:
                prompt += " Choose the next city to open a branch in: {}.".format(", ".join(valid_nodes))

        return prompt

    def _get_valid_nodes(self, next_node, visited_nodes):
        valid_nodes = set(
            sum(
                [(self._get_adj_nodes(node) + [node]) for node in visited_nodes + [next_node]],
                []
            )
        )
        assert self._start_node in valid_nodes

        return valid_nodes

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
