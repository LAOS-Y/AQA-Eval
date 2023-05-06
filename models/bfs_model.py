import re

def extract_int(s):
    def isint(word):
        try:
            int(word)
            return True
        except ValueError:
            return False

    return [int(word) for word in re.split("\n|,| |\t|\(|\)|\[|\]|\.", s) if isint(word)]

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
        self.adjacency = extract_int(prompt[prompt.rindex("["):prompt.rindex("]") + 1])

class BFSModel:

    def __init__(self):
        self.reset("")

    def reset(self, instruction):
        self.visited = {0}
        self.history = [{0}]
        self.node_stack = [{0}]  # dfs trajectory

    def __call__(self, prompt):
        input = ModelInput(prompt)
        adjacency = input.adjacency

        next_adjacency = set()
        for adj_node in adjacency:
            if adj_node not in self.visited:
                next_adjacency.add(adj_node)
                self.visited.add(adj_node)

        if next_adjacency:
            self.history.append(next_adjacency)
            self.node_stack.append(next_adjacency)
            return str(next_adjacency)

        if len(self.node_stack) == 0:
            return "over"

    def force(self, new_reply):
        self.history[-1] = new_reply
        self.node_stack[-1] = new_reply
