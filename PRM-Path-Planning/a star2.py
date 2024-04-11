import heapq
import matplotlib.pyplot as plt

class Graph:
    def __init__(self):
        self.edges = {}
    
    def add_edge(self, node1, node2, cost):
        if node1 not in self.edges:
            self.edges[node1] = {}
        self.edges[node1][node2] = cost
        
        if node2 not in self.edges:
            self.edges[node2] = {}
        self.edges[node2][node1] = cost

def heuristic(node, goal):
    # 使用曼哈顿距离作为启发式函数
    x1, y1 = node
    x2, y2 = goal
    return abs(x1 - x2) + abs(y1 - y2)

def astar(graph, start, goal):
    open_set = [] # 待考察集合，用优先队列表示
    closed_set = set() # 已探索集合
    heapq.heappush(open_set, (0, start)) # 初始化考察起点
    
    came_from = {}
    g_score = {start: 0} # 字典存储 节点：cost
    f_score = {start: heuristic(start, goal)}
    
    while open_set:
        _, current = heapq.heappop(open_set) # 2. 弹出最小cost的备选子节点
        print("Selected lowest cost Node:", current) 
        
        if current == goal:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            return path[::-1], came_from  # 返回路径和父节点关系
        
        closed_set.add(current) # 已被考察
        
        for neighbor, cost in graph.edges[current].items():
            print("Current Node:", current, "Neighbor Node:", neighbor) # 打印当前节点和邻居节点
            
            if neighbor in closed_set:
                continue
            
            tentative_g_score = g_score[current] + cost
            if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                came_from[neighbor] = current  # 更新父节点关系
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                heapq.heappush(open_set, (f_score[neighbor], neighbor)) # 1. 计算所有childen的 cost，入选 openlist
    
    return None, None

def plot_graph(graph, path=None, came_from=None):
    for node, neighbors in graph.edges.items():
        plt.scatter(node[0], node[1], color='blue')  # 绘制节点
        plt.text(node[0], node[1], f"{node}", fontsize=8, ha='right', va='bottom')  # 文字描述节点
        
        for neighbor in neighbors:
            plt.plot([node[0], neighbor[0]], [node[1], neighbor[1]], color='gray')  # 绘制边
    
    if path:
        for i in range(len(path)-1):
            plt.plot([path[i][0], path[i+1][0]], [path[i][1], path[i+1][1]], color='red')  # 绘制路径
    
    if came_from:
        for child, parent in came_from.items():
            if parent:
                plt.plot([parent[0], child[0]], [parent[1], child[1]], color='green', linestyle='--')  # 绘制父节点关系
    
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('A* Algorithm Visualization')
    plt.grid()
    plt.show()

# 示例图
graph = Graph()
graph.add_edge((0, 0), (1, 0), 1)  # A
graph.add_edge((1, 0), (1, 1), 1)  # B
graph.add_edge((1, 1), (2, 1), 1)  # C
graph.add_edge((2, 1), (2, 2), 1)  # D
graph.add_edge((2, 2), (3, 2), 1)  # E
graph.add_edge((3, 2), (3, 3), 1)  # F
graph.add_edge((0, 0), (0, 1), 1)  # G
graph.add_edge((0, 1), (1, 1), 1)  # H
graph.add_edge((1, 1), (1, 2), 1)  # I
graph.add_edge((1, 2), (2, 2), 1)  # J
graph.add_edge((2, 2), (2, 3), 1)  # K
graph.add_edge((2, 3), (3, 3), 1)  # L
graph.add_edge((0, 1), (0, 2), 1)  # M
graph.add_edge((0, 2), (1, 2), 1)  # N
graph.add_edge((1, 0), (2, 0), 1)  # O
graph.add_edge((2, 0), (2, 1), 1)  # P

start = (0, 0)
goal = (3, 3)

# 执行A*算法
path, came_from = astar(graph, start, goal)
print("Path:", path)
print("come from:", came_from)
# 绘制图形
plot_graph(graph, path, came_from)

