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
    explored_nodes = {}  # Dictionary to store explored nodes in tree structure
    
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
            return path, explored_nodes  # Return explored nodes along with the path
        
        closed_set.add(current_state) # 已访问
        
        if current_node.parent:
            parent_state = current_node.parent.state
            if parent_state not in explored_nodes:
                explored_nodes[parent_state] = []
            explored_nodes[parent_state].append(current_state)
        else:
            explored_nodes[current_state] = []
        
        for next_state, cost in successors(current_state): # 访问相邻子节点
            if next_state in closed_set:
                continue
            
            g = current_node.g + cost
            h = heuristic(next_state, goal_state)
            next_node = Node(next_state, current_node, g, h) # 子节点
            
            if next_node in open_list:
                continue
            
            heapq.heappush(open_list, next_node) #所有子节点都进待选队列
    
    return None, explored_nodes  # Return explored nodes if goal state not found

# 示例：八数码问题
def successors(state):
    # 获取当前状态的后继状态及其转移代价
    successors = []
    for move in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
        new_state = (state[0] + move[0], state[1] + move[1])
        if 0 <= new_state[0] < 3 and 0 <= new_state[1] < 3:
            successors.append((new_state, 1))  # 四个方向的移动代价均为1
    return successors

# 递归绘制树
def draw_tree(G, pos, node, explored_nodes, level):
    children = explored_nodes.get(node, [])
    if children:
        for child in children:
            plt.plot([pos[node][0], pos[child][0]], [pos[node][1], pos[child][1]], 'b-')  # 绘制边
            draw_tree(G, pos, child, explored_nodes, level + 1)

# 绘制路径
def plot_path(G, path, explored_nodes):
    pos = nx.spring_layout(G)  # 重新计算节点位置

    # 绘制已探索的节点
    for node in explored_nodes.keys():
        nx.draw_networkx_nodes(G, pos, nodelist=[node], node_color='blue', node_size=500)

    # 绘制路径
    for i in range(len(path) - 1):
        plt.plot([pos[path[i]][0], pos[path[i + 1]][0]], [pos[path[i]][1], pos[path[i + 1]][1]], 'r-')  # 绘制路径边

    # 绘制树形结构
    draw_tree(G, pos, path[-1], explored_nodes, 1)

    # 显示图形
    plt.show()

start_state = (0, 0)
goal_state = (2, 2)
path, explored_nodes = astar(start_state, goal_state, successors)
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
plot_path(G, path, explored_nodes)
