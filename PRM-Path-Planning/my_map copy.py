import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict

class Vertex:
    """顶点类"""
    def __init__(self, val):
        self.val = val

class GraphAdjList:
    """基于邻接表实现的无向图类"""

    def __init__(self, edges):
        """构造方法"""
        self.adj_list = defaultdict(list)
        self._build_graph(edges)

    def _build_graph(self, edges):
        """构建图"""
        for edge in edges:
            self.add_edge(edge[0], edge[1])

    def size(self):
        """获取顶点数量"""
        return len(self.adj_list)

    def add_edge(self, vertex1, vertex2):
        """添加边"""
        if vertex1 == vertex2:
            raise ValueError("Self loops are not allowed.")
        self.adj_list[vertex1].append(vertex2)
        self.adj_list[vertex2].append(vertex1)

    def remove_edge(self, vertex1, vertex2):
        """删除边"""
        if vertex1 not in self.adj_list or vertex2 not in self.adj_list:
            raise ValueError("One or both vertices not in graph.")
        self.adj_list[vertex1].remove(vertex2)
        self.adj_list[vertex2].remove(vertex1)

    def add_vertex(self, vertex):
        """添加顶点"""
        if vertex not in self.adj_list:
            self.adj_list[vertex] = []

    def remove_vertex(self, vertex):
        """删除顶点"""
        if vertex not in self.adj_list:
            raise ValueError("Vertex not in graph.")
        neighbors = self.adj_list.pop(vertex)
        for neighbor in neighbors:
            self.adj_list[neighbor].remove(vertex)

    def visualize(self):
        """可视化图"""
        G = nx.Graph()
        for vertex, neighbors in self.adj_list.items():
            G.add_node(vertex.val)
            for neighbor in neighbors:
                G.add_edge(vertex.val, neighbor.val)
        nx.draw(G, with_labels=True, font_weight='bold')
        plt.show()

    def print(self):
        """打印邻接表"""
        print("邻接表 =")
        for vertex, neighbors in self.adj_list.items():
            neighbor_vals = [neighbor.val for neighbor in neighbors]
            print(f"{vertex.val}: {neighbor_vals}")



# 示例用法
vertices = ['1', '3', '2', '5','4']
edges = [(0, 1), (0, 3), (1, 0), (1, 2),(2,1),(2,3),(2,4),(3,0),(3,2),(3,4),(4,2),(4,3)]

graph = GraphAdjMat(vertices, edges)
graph.print()
graph.visualize()
