import random
import networkx 
import re
from loguru import logger
from utils import DialogLogger
from models.dfs_model import DFSModel

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
    def __init__(self) -> None:
        self.reset()
    
    def reset(self):
        self.dialog_logger = DialogLogger(order=["Q", "A", "T"])
        self.teacher = DFSModel()
    
    @property
    def default_insturction(self):
        return "You are on a node of an undirected non-cyclic graph. "\
                "An undirected non-cyclic garph contains a set of node, and a set of edges that each connects a pair of nodes. All edges are undirected, so that you can move from one node to the other connected by the edge in either direction. \n" \
                "You are asked to visit all nodes in the graph. \n" \
                "Every time you enter a node, you will be given the adjacent nodes connected to this node. \n" \
                "You can move to a node by responding the node ID adjacent to the node. \n" \
                "The game will finish once you have visited all the nodes in the graph. \n"

    def _get_adj_nodes(self, graph, curr_node):
        return [n for _, n in graph.edges(curr_node)]
    
    def _generate_exploring_prompt(self, 
                                   graph, 
                                   curr_node,
                                   node_history,
                                   mcq,
                                   provide_state):
        '''
        Generate prompt used in exploration step

        Return: prompt (string)
        '''

        adj_nodes = self._get_adj_nodes(graph, curr_node)

        prompt = "You are on node {}, " \
                 "number of adjacent node is {}, " \
                 "adjacent nodes are {}".format(
                    str(curr_node),
                    len(adj_nodes), 
                    ', '.join([str(i) for i in adj_nodes]))

        if provide_state:
            unused_nodes = set(adj_nodes).difference(set(node_history))
            if len(unused_nodes) == 0:
                prompt += " You have visited all nodes adjacent to this node."
            else:
                prompt += " You have not visited node {} adjacent to this node.".format(', '.join([str(i) for i in unused_nodes]))
        if mcq:
            prompt += " Select the next node ID from the following selections: {}".format(', '.join([str(i) for i in adj_nodes]))

        return prompt


    def _explore_graph_step(self, 
                            graph, curr_node,
                            model_response, # model/teacher string response
                            node_history, # variables for tracking node states
                            mcq, provide_state # model evaluation config
                            ):
        '''
        Process model response, decide prompt used in next step or finish exploring
        
        Return 
        - string: prompt used in the next exploring step
        - int: next node
        - boolean: if selected interface is same as teacher model
        - float: coverage of nodes after exploration
        '''

        ints = extract_int(model_response)
        if len(ints) != 1:
            raise RuntimeError("Returning int number not equal to 1")
        
        next_node = ints[0]
        adj_nodes = self._get_adj_nodes(graph, curr_node)
        if next_node not in adj_nodes:
            raise RuntimeError("Selected next node not in adjacent node")
        
        # check if model selected node following dfs path. i.e. select unvisited child node or parent node
        unused_nodes = set(adj_nodes).difference(set(node_history))
        if len(unused_nodes) == 0:
            # if all child have been fisited, check if model is visiting its parent node in the history stack
            parent_node_index = node_history.index(curr_node) - 1
            if parent_node_index < 0:
                # if curr node is root, fail this correctness check
                dfs_correctness = False
            else:
                dfs_correctness = (next_node == node_history[parent_node_index])
        else:
            # should visit child node
            dfs_correctness = (next_node in unused_nodes)

        prompt = self._generate_exploring_prompt(graph, next_node, node_history, mcq, provide_state)
        
        node_history.append(next_node)

        cov = len(set(node_history)) / len(graph.nodes)

        return prompt, next_node, dfs_correctness, cov


    def init_model(self, model, teacher_forcing=False, explain_algo=False):
        
        model.reset()
        prompt = self.default_insturction

        if explain_algo:
            prompt += "You should use depth first search algorithm, each time you should select a node you have not moved to. If all nodes adjacent to the current node have been visited, you should back track to the node through which you entered this node. "

        prompt += "Try move as few times as you can.\n" \
                  "Reply 'OK' if you understand."
        
        self.dialog_logger.info(Q=prompt)
        self.teacher(prompt=prompt)

        reply = model(prompt).lower()
        if teacher_forcing:
            self.dialog_logger.info(A=reply, T="OK")
            model.force("OK")
            return True
        
        self.dialog_logger.info(A=reply)
        return "ok" in reply

    def test_one_time(self, model, teacher_forcing, mcq, explain_algo, provide_state):

        self.reset()
        if not self.init_model(model, teacher_forcing, explain_algo):
            raise RuntimeError("failed to init model")
        
        # info required for recording and iterative eval
        cnt = 0
        # node_num = random.randint(3, 4)
        node_num = 10
        graph = networkx.random_tree(node_num).to_undirected()
        curr_node = random.randint(0, node_num-1)
        
        prompt = "START. " + self._generate_exploring_prompt(graph, curr_node, [], mcq, provide_state)
        
        retry_cnt = 0

        node_history = [curr_node]
        
        logger.info("Generated random graph: nodes: {}, edges: {}".format(graph.nodes, graph.edges))

        correct_cnt = 0
        cov_sum = 0
        cov = 0

        while cnt < 20 and retry_cnt <= 3:
            self.dialog_logger.info(Q=prompt)
            
            model_response = model(prompt).lower()

            if not teacher_forcing:
                self.dialog_logger.info(A=model_response)
                teacher_response = ""
            else:
                teacher_response = self.teacher(prompt=prompt)
                self.dialog_logger.info(A=model_response, T=teacher_response)
                model.force(teacher_response)

            # start processing response in this iteration
            try:
                # if all node visited, finish 
                if cov == 1.0:
                    break
                
                # if ground truth has finished, end evaluation
                if teacher_forcing and teacher_response == "null":
                    break

                prompt, curr_node, dfs_correctness, cov = self._explore_graph_step(graph, curr_node,
                                                            model_response, node_history, 
                                                            mcq, provide_state)
                
                if dfs_correctness:
                    correct_cnt += 1
                cov_sum += 1 - cov

                cnt += 1
                retry_cnt = 0
            except Exception as e:
                print(e)
                if retry_cnt == 0:
                    prompt = "Invalid response. Try again. Please do not include any reasoning in your response. " + prompt
                retry_cnt += 1

        print(f"finished, cnt: {cnt}")
        print(f"nodes: {graph.nodes}, edges: {graph.edges}")
        print(f"history: {node_history}")
        print(f"coverage: sum: {cov_sum}, min: {1 - cov}")
        print(f"accuracy: {correct_cnt / cnt}")

        return cnt, cov_sum, cov, correct_cnt / cnt

