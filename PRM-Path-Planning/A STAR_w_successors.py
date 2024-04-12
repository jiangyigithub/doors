import heapq
import networkx as nx
import matplotlib.pyplot as plt

class Node:
    def __init__(self, state, parent=None, g=0, h=0):
        self.state = state  # 当前节点所在位置
        self.parent = parent  # 父节点
        self.g = g  # 从起始节点到当前节点的实际代价
        self.h = h  # 从当前节点到目标节点的启发式估计代价（也称为启发式函数值）

# 优先级队列的比较方式
    def __lt__(self, other):
        return (self.g + self.h) < (other.g + other.h)

def heuristic(state, goal_state):
    # 曼哈顿距离启发式函数
    return abs(state[0] - goal_state[0]) + abs(state[1] - goal_state[1])

def astar(start_state, goal_state, successors):
    open_list = []
    closed_set = set()
    
    start_node = Node(start_state, None, 0, heuristic(start_state, goal_state))
    heapq.heappush(open_list, start_node)
    
    while open_list:
        current_node = heapq.heappop(open_list) # 取第一个节点
        current_state = current_node.state
        
        if current_state == goal_state:
            path = []
            while current_node:
                path.append(current_node.state)
                current_node = current_node.parent
            path.reverse()
            return path
        
        closed_set.add(current_state) # 已访问
        
        for next_state, cost in successors(current_state): # 访问相邻子节点
            if next_state in closed_set:
                continue
            
            g = current_node.g + cost
            h = heuristic(next_state, goal_state)
            next_node = Node(next_state, current_node, g, h) # 子节点
            
            if next_node in open_list:
                continue
            
            heapq.heappush(open_list, next_node) #所有子节点都进待选队列
    
    return None

# 示例：八数码问题
def successors(state):
    # 获取当前状态的后继状态及其转移代价
    successors = []
    for move in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
        new_state = (state[0] + move[0], state[1] + move[1])
        if 0 <= new_state[0] < 3 and 0 <= new_state[1] < 3:
            successors.append((new_state, 1))  # 四个方向的移动代价均为1
    return successors

# 绘制路径
def plot_path(G, path):
    pos = {node: node for node in G.nodes()}  # 节点位置
    edge_labels = {(u, v): d['weight'] for u, v, d in G.edges(data=True)}  # 边权重
    nx.draw(G, pos, with_labels=True, node_size=700, node_color='lightblue')  # 绘制节点
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)  # 绘制边权重
    path_edges = [(path[i], path[i+1]) for i in range(len(path)-1)]  # 路径边集合
    nx.draw_networkx_edges(G, pos, edgelist=path_edges, edge_color='red', width=2.0)  # 绘制路径
    plt.show()

start_state = (0, 0)
goal_state = (2, 2)
path = astar(start_state, goal_state, successors)
print("A*算法找到的最短路径：", path)

# 创建有向图
G = nx.DiGraph()

# 添加节点和边
for i in range(3):
    for j in range(3):
        G.add_node((i, j))
for i in range(3):
    for j in range(3):
        for move in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            new_state = (i + move[0], j + move[1])
            if 0 <= new_state[0] < 3 and 0 <= new_state[1] < 3:
                G.add_edge((i, j), new_state, weight=1)

# 绘制路径
plot_path(G, path)
