import numpy as np
import random
from collections import defaultdict
import matplotlib.pyplot as plt
import networkx as nx

class POMDP:
    def __init__(self, states, actions, observations, transition_probs, observation_probs, rewards, gamma):
        self.states = states
        self.actions = actions
        self.observations = observations
        self.transition_probs = transition_probs
        self.observation_probs = observation_probs
        self.rewards = rewards
        self.gamma = gamma

    def transition(self, state, action):
        return np.random.choice(self.states, p=self.transition_probs[(state, action)])

    def observe(self, state, action):
        return np.random.choice(self.observations, p=self.observation_probs[(state, action)])

    def reward(self, state, action):
        return self.rewards[(state, action)]

class MCTSNode:
    def __init__(self, belief, parent=None, action=None):
        self.belief = belief
        self.parent = parent
        self.action = action
        self.children = defaultdict(MCTSNode)  # 使用defaultdict
        self.visits = 0
        self.value = 0.0

def belief_update(belief, action, observation, pomdp):
    new_belief = np.zeros(len(pomdp.states))
    for s in pomdp.states:
        for sp in pomdp.states:
            new_belief[sp] += belief[s] * pomdp.transition_probs[(s, action)][sp] * pomdp.observation_probs[(sp, action)][observation]
    new_belief /= np.sum(new_belief)
    return new_belief

def select_action(node, pomdp):
    if not node.children:
        return pomdp.actions[0]  # 假设返回第一个动作
    return max(node.children, key=lambda a: (node.children[a].value / (node.children[a].visits + 1e-6)))

def simulate(node, pomdp, depth):
    if depth == 0:
        return pomdp.reward(np.random.choice(pomdp.states, p=node.belief), None)

    action = select_action(node, pomdp)
    next_node = node.children[action]

    next_state = pomdp.transition(pomdp.states[np.argmax(next_node.belief)], action)
    reward = pomdp.reward(next_state, action)
    observation = pomdp.observe(next_state, action)
    new_belief = belief_update(next_node.belief, action, observation, pomdp)
    child_node = MCTSNode(new_belief, next_node, observation)  # 更新new_belief的使用

    q_value = reward + pomdp.gamma * simulate(child_node, pomdp, depth - 1)
    next_node.visits += 1
    next_node.value += q_value

    return q_value

def mcts(pomdp, initial_belief, iterations, depth):
    root = MCTSNode(initial_belief)
    for _ in range(iterations):
        simulate(root, pomdp, depth)
    best_action = max(root.children, key=lambda a: root.children[a].value / root.children[a].visits)
    return best_action

# Example usage
states = [0, 1, 2]
actions = [0, 1]
observations = [0, 1]
transition_probs = {
    (0, 0): [0.7, 0.2, 0.1],
    (0, 1): [0.1, 0.6, 0.3],
    (1, 0): [0.3, 0.4, 0.3],
    (1, 1): [0.4, 0.4, 0.2],
    (2, 0): [0.5, 0.3, 0.2],
    (2, 1): [0.2, 0.3, 0.5],
}
observation_probs = {
    (0, 0): [0.9, 0.1],
    (0, 1): [0.7, 0.3],
    (1, 0): [0.6, 0.4],
    (1, 1): [0.8, 0.2],
    (2, 0): [0.4, 0.6],
    (2, 1): [0.5, 0.5],
}
rewards = {
    (0, 0): 5,
    (0, 1): 10,
    (1, 0): -1,
    (1, 1): 2,
    (2, 0): 1,
    (2, 1): 3,
}
gamma = 0.9

pomdp = POMDP(states, actions, observations, transition_probs, observation_probs, rewards, gamma)
initial_belief = np.array([0.4, 0.4, 0.2])


# 创建图形
def plot_pomdp(pomdp):
    G = nx.MultiDiGraph()

    # 添加节点
    for state in pomdp.states:
        G.add_node(state, label=f'State {state}')
    
    # 添加边（带有动作和转移概率）
    for (state, action), probs in pomdp.transition_probs.items():
        for next_state, prob in enumerate(probs):
            if prob > 0:
                G.add_edge(state, next_state, label=f'A{action}, P={prob:.2f}', action=action, weight=prob)

    pos = nx.spring_layout(G)

    plt.figure(figsize=(12, 8))
    nx.draw(G, pos, with_labels=True, node_size=3000, node_color='skyblue', font_size=15, font_weight='bold', arrowsize=20, connectionstyle='arc3,rad=0.1')
    edge_labels = {(u, v): d['label'] for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=10, font_color='red')

    # 标注奖励
    for (state, action), reward in pomdp.rewards.items():
        x, y = pos[state]
        plt.text(x, y - 0.1, f'R{action}={reward}', fontsize=10, ha='center')

    plt.title('POMDP State Transition Diagram')
    plt.show()

plot_pomdp(pomdp)

# best_action = mcts(pomdp, initial_belief, iterations=1000, depth=10)
# print("Best action:", best_action)
