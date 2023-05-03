import re

def extract_int(s):
  def isint(word):
    try:
      int(word)
      return True
    except ValueError:
      return False
  return [int(word) for word in re.split("\n|,| |\t|\(|\)|\[|\]", s) if isint(word)]

CURR_NODE_IDX = 0
ADJ_NODE_NUM_IDX = 1
ADJ_NODE_IDX = 2

class DFSModel():
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.node_history = []
        self.node_stack = [] # list of parent nodes
        self.last_response = ""

    def __call__(self, prompt):
        prompt = prompt.lower()
        if "try again" in prompt:
            return self.last_response
        if "ok" in prompt:
            self.reset()
            self.last_response = "OK"
            return "OK"
                
        if "start" in prompt:
            info = extract_int(prompt)
            self.node_history.append(info[CURR_NODE_IDX])
            self.node_stack.append(info[CURR_NODE_IDX])
            
            self.last_response = str(info[ADJ_NODE_IDX])
            return self.last_response
        else:            
            info = extract_int(prompt)
            self.node_history.append(info[CURR_NODE_IDX])

            # select a node not fisited before
            for next_node in info[ADJ_NODE_IDX: ADJ_NODE_IDX + info[ADJ_NODE_NUM_IDX]]:
                if next_node not in self.node_history:
                    self.node_stack.append(info[CURR_NODE_IDX])
                    self.last_response = str(next_node)
                    return self.last_response

            # backtrack to parent node
            # if no node exist, end exploring
            if len(self.node_stack) == 0:
               return "null"
            self.last_response = str(self.node_stack[-1])
            self.node_stack.pop() # pop parent node
            return self.last_response
        
    def force(self, new_reply):
        return