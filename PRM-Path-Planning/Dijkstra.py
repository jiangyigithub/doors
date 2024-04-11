import heapq
import networkx as nx
import matplotlib.pyplot as plt

def dijkstra(graph, start):
    # 初始化距离字典和前驱节点字典
    distance = {node: float('inf') for node in graph}
    distance[start] = 0
    predecessor = {}
    
    # 初始化优先队列
    priority_queue = [(0, start)]
    
    while priority_queue:
        # 从优先队列中取出距离最小的节点
        current_distance, current_node = heapq.heappop(priority_queue)
        
        # 如果当前节点已经访问过，则跳过
        if current_distance > distance[current_node]:
            continue
        
        # 遍历当前节点的邻居节点
        for neighbor, weight in graph[current_node].items():
            distance_to_neighbor = current_distance + weight
            
            # 如果通过当前节点到达邻居节点的路径距离小于目前已知的距离，则更新距离
            if distance_to_neighbor < distance[neighbor]:
                distance[neighbor] = distance_to_neighbor
                predecessor[neighbor] = current_node
                # 将邻居节点加入优先队列
                heapq.heappush(priority_queue, (distance_to_neighbor, neighbor))
    
    return distance, predecessor

# 示例图的邻接列表表示
graph = {
    'A': {'B': 5, 'C': 3},
    'B': {'A': 5, 'C': 1, 'D': 3},
    'C': {'A': 3, 'B': 1, 'D': 2},
    'D': {'B': 3, 'C': 2}
}

# 在节点A开始执行Dijkstra算法
start_node = 'A'
distances, predecessors = dijkstra(graph, start_node)

# 输出每个节点的最短路径和路径长度
for node in graph:
    path = []
    current_node = node
    while current_node != start_node:
        path.append(current_node)
        current_node = predecessors[current_node]
    path.append(start_node)
    path.reverse()
    print(f"最短路径到达节点 {node}: {' -> '.join(path)}, 路径长度为 {distances[node]}")

# 创建有向图
G = nx.DiGraph()

# 添加节点和边
for node, neighbors in graph.items():
    for neighbor, weight in neighbors.items():
        G.add_edge(node, neighbor, weight=weight)

# 绘制图形
pos = nx.spring_layout(G)  # 设置节点位置
nx.draw(G, pos, with_labels=True, node_size=700, node_color='lightblue')  # 绘制节点
nx.draw_networkx_edge_labels(G, pos, edge_labels={(u, v): d['weight'] for u, v, d in G.edges(data=True)})  # 绘制边权重
plt.show()

