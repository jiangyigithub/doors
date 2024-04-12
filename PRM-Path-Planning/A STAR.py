import heapq
import networkx as nx
import matplotlib.pyplot as plt

class Node:
    def __init__(self, state, parent=None, g=0, h=0):
        self.state = state
        self.parent = parent
        self.g = g
        self.h = h

    def __lt__(self, other):
        return (self.g + self.h) < (other.g + other.h)

def heuristic(state, goal_state):
    return abs(state[0] - goal_state[0]) + abs(state[1] - goal_state[1])

def astar(start_state, goal_state, graph):
    open_list = []
    closed_set = set()
    
    start_node = Node(start_state, None, 0, heuristic(start_state, goal_state))
    heapq.heappush(open_list, start_node)
    
    while open_list:
        current_node = heapq.heappop(open_list)
        current_state = current_node.state
        
        if current_state == goal_state:
            path = []
            while current_node:
                path.append(current_node.state)
                current_node = current_node.parent
            path.reverse()
            return path
        
        closed_set.add(current_state)
        
        for neighbor in graph.neighbors(current_state):
            next_state = neighbor
            cost = graph[current_state][neighbor]['weight']
            
            if next_state in closed_set:
                continue
            
            g = current_node.g + cost
            h = heuristic(next_state, goal_state)
            next_node = Node(next_state, current_node, g, h)
            
            if next_node in open_list:
                continue
            
            heapq.heappush(open_list, next_node)
    
    return None

def plot_path(G, path):
    pos = {node: node for node in G.nodes()}  # 节点位置
    edge_labels = {(u, v): d['weight'] for u, v, d in G.edges(data=True)}  # 边权重
    nx.draw(G, pos, with_labels=True, node_size=700, node_color='lightblue')  # 绘制节点
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)  # 绘制边权重
    path_edges = [(path[i], path[i+1]) for i in range(len(path)-1)]  # 路径边集合
    nx.draw_networkx_edges(G, pos, edgelist=path_edges, edge_color='red', width=2.0)  # 绘制路径
    plt.show()

# 创建有向图
G = nx.DiGraph()

# 添加节点和边
for i in range(4):
    for j in range(4):
        G.add_node((i, j))

# 添加水平和垂直边
for i in range(4):
    for j in range(3):
        G.add_edge((i, j), (i, j+1), weight=1)
        G.add_edge((j, i), (j+1, i), weight=1)

# 添加斜对角边
for i in range(3):
    for j in range(3):
        G.add_edge((i, j), (i+1, j+1), weight=2)
        G.add_edge((i+1, j), (i, j+1), weight=2)

# 指定起始和目标节点
start_state = (0, 0)
goal_state = (2, 3)

# 绘制图形
# pos = {(x, y): (y, -x) for x in range(4) for y in range(4)}  # 位置信息
# nx.draw(G, pos, with_labels=True, node_size=700, node_color='lightblue')  # 绘制节点
# plt.show()
path = astar(start_state, goal_state, G)
print("A*算法找到的最短路径：", path)

# 绘制路径
plot_path(G, path)
