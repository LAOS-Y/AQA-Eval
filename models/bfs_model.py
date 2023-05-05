import re
from collections import deque

def extract_int(s):
    def isint(word):
        try:
            int(word)
            return True
        except ValueError:
            return False

    return [int(word) for word in re.split("\n|,| |\t|\(|\)|\[|\]", s) if isint(word)]

class ModelInput:
    def __init__(self, prompt):
        self.cur_node = 0
        self.adjacency = []
        self.parse(prompt)

    def parse(self, prompt):
        if 'start node' in prompt:
            raw = extract_int(prompt[prompt.index('start node'):])
            if len(raw) <= 1:
                return
            self.cur_node = raw[0]
        # self.adjacency = extract_int(prompt[prompt.index('Adjacent'):])


class BFSModel:

    def __init__(self) -> object:
        self.reset()

    def reset(self):
        self.adjacency_list = {}
        self.visited = {}
        self.distance = {}
        self.path = {}
        self.node_history = []
        self.queue = deque()
        self.added = {}

    def add_edge(self, cur_node, adjacency):
        if cur_node not in self.adjacency_list.keys():
            self.adjacency_list[cur_node] = []

        for node in adjacency:
            if node not in self.adjacency_list[cur_node]:
                self.adjacency_list[cur_node].append(node)

            if node not in self.adjacency_list.keys():
                self.adjacency_list[node] = []
            if cur_node not in self.adjacency_list[node]:
                self.adjacency_list[node].append(cur_node)

    def add_node(self, cur_node):
        if cur_node in self.added.keys() and self.added[cur_node]:
            return
        if cur_node not in self.visited:
            self.visited[cur_node] = False
        if cur_node not in self.distance:
            self.distance[cur_node] = 0
        if cur_node not in self.path:
            self.path[cur_node] = []
        if cur_node not in self.added:
            self.added[cur_node] = True

    def __call__(self, prompt):
        if "ok" in prompt:
            self.reset()
            return "ok"

        input = ModelInput(prompt)
        if "START" in prompt:
            self.visited[input.cur_node] = True
            self.distance[input.cur_node] = 0
            self.path[input.cur_node] = [input.cur_node]
            self.queue.append(input.cur_node)

        current_node = self.queue.popleft()
        self.node_history.append(current_node)

        adjacency = input.adjacency
        self.add_edge(current_node, adjacency)
        self.add_node(current_node)
        for node in adjacency:
            self.add_node(node)

        for neighbor in self.adjacency_list[current_node]:
            if not self.visited[neighbor]:
                self.visited[neighbor] = True
                self.distance[neighbor] = self.distance[current_node] + 1
                self.path[neighbor] = self.path[current_node] + [neighbor]
                self.queue.append(neighbor)

        if self.queue:
            return "move to {}".format(self.queue[0])

        self.distance = {tup[0]: tup[1] for tup in sorted(self.distance.items())}
        return f'over. distance: {self.distance}\n'

    def teacher_force(self, new_reply):
        return  # optimal model does not need teacher forcing
