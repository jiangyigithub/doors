
import networkx as nx
import matplotlib.pyplot as plt

def print_matrix(matrix):
    for row in matrix:
        print(" ".join(str(cell) for cell in row))

class GraphAdjMat:
    """基于邻接矩阵实现的无向图类"""

    def __init__(self, vertices, edges):
        """构造方法"""
        # 顶点列表，元素代表“顶点值”，索引代表“顶点索引”
        self.vertices = []
        # 邻接矩阵，行列索引对应“顶点索引”
        self.adj_mat = []
        # 添加顶点
        for val in vertices:
            self.add_vertex(val)
        # 添加边
        # 请注意，edges 元素代表顶点索引，即对应 vertices 元素索引
        for e in edges:
            self.add_edge(e[0], e[1])

    def size(self):
        """获取顶点数量"""
        return len(self.vertices)

    def add_vertex(self, val):
        """添加顶点"""
        n = self.size()
        # 向顶点列表中添加新顶点的值
        self.vertices.append(val)
        # 在邻接矩阵中添加一行
        new_row = [0] * n
        self.adj_mat.append(new_row)
        # 在邻接矩阵中添加一列
        for row in self.adj_mat:
            row.append(0)

    def remove_vertex(self, index):
        """删除顶点"""
        if index >= self.size():
            raise IndexError()
        # 在顶点列表中移除索引 index 的顶点
        self.vertices.pop(index)
        # 在邻接矩阵中删除索引 index 的行
        self.adj_mat.pop(index)
        # 在邻接矩阵中删除索引 index 的列
        for row in self.adj_mat:
            row.pop(index)

    def add_edge(self, i, j):
        """添加边"""
        # 参数 i, j 对应 vertices 元素索引
        # 索引越界与相等处理
        if i < 0 or j < 0 or i >= self.size() or j >= self.size() or i == j:
            raise IndexError()
        # 在无向图中，邻接矩阵关于主对角线对称，即满足 (i, j) == (j, i)
        self.adj_mat[i][j] = 1
        self.adj_mat[j][i] = 1

    def remove_edge(self, i, j):
        """删除边"""
        # 参数 i, j 对应 vertices 元素索引
        # 索引越界与相等处理
        if i < 0 or j < 0 or i >= self.size() or j >= self.size() or i == j:
            raise IndexError()
        self.adj_mat[i][j] = 0
        self.adj_mat[j][i] = 0

    def visualize(self):
        """可视化图"""
        G = nx.Graph()
        for i, vertex in enumerate(self.vertices):
            G.add_node(i, label=str(vertex))
        for i in range(self.size()):
            for j in range(i+1, self.size()):
                if self.adj_mat[i][j] == 1:
                    G.add_edge(i, j)
        labels = nx.get_node_attributes(G, 'label')
        pos = nx.spring_layout(G)
        nx.draw(G, pos, with_labels=True, labels=labels)
        plt.show()

    def print(self):
        """打印邻接矩阵"""
        print("顶点列表 =", self.vertices)
        print("邻接矩阵 =")
        for row in self.adj_mat:
            print(" ".join(str(cell) for cell in row))


# 示例用法
vertices = ['A', 'B', 'C', 'D']
edges = [(0, 1), (0, 2), (1, 2), (2, 3)]

graph = GraphAdjMat(vertices, edges)
graph.print()
graph.visualize()
