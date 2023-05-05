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
        print(prompt[prompt.rindex("["):])
        self.adjacency = extract_int(prompt[prompt.rindex("["):])


# prompt = "'START. The start node is 4. Adjacent nodes of node4 are [3]'"
# model_input = ModelInput(prompt)
# tmp = re.split("\n|,| |\t|\(|\)|\[|\]|\.", prompt)
# print(tmp)
class BFSModel:

    def __init__(self):
        self.visited = set()

    def reset(self):
        self.visited = set()

    def __call__(self, prompt):
        if "ok" in prompt:
            self.reset()
            return "ok"

        input = ModelInput(prompt)
        if "START" in prompt:
            self.visited.add(input.cur_node)

        adjacency = input.adjacency

        print("teacher adjacency: ", adjacency)

        next_adjacency = set()
        for adj_node in adjacency:
            if adj_node not in self.visited:
                next_adjacency.add(adj_node)
                self.visited.add(adj_node)

        if next_adjacency:
            return "Next i will traverse nodes {}".format(next_adjacency)

        return f'over\n'

    def teacher_force(self, new_reply):
        return  # optimal model does not need teacher forcing
