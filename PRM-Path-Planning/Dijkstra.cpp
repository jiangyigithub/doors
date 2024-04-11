#include <iostream>
#include <vector>
#include <queue>
#include <limits>

using namespace std;

#define INF numeric_limits<int>::max()

// 边的结构体
struct Edge {
    int to;
    int cost;
};

// 图的结构体
struct Graph {
    vector<vector<Edge>> adj;
    int V; // 顶点数

    // 构造函数
    Graph(int V) : V(V), adj(V) {}

    // 添加边的函数
    void addEdge(int u, int v, int cost) {
        adj[u].push_back({v, cost});
    }
};

// Dijkstra算法实现
vector<int> dijkstra(const Graph& graph, int start) {
    vector<int> dist(graph.V, INF); // 距离数组，初始化为无穷大
    dist[start] = 0; // 起始点到自身的距离为0

    priority_queue<pair<int, int>, vector<pair<int, int>>, greater<pair<int, int>>> pq; // 最小堆，存储（距离，顶点）对
    pq.push({0, start}); // 将起始点加入最小堆

    while (!pq.empty()) {
        int d = pq.top().first;
        int u = pq.top().second;
        pq.pop();

        // 如果从起始点到顶点u的距离大于当前最小距离，则跳过
        if (d > dist[u])
            continue;

        // 遍历u的所有邻居节点
        for (const Edge& e : graph.adj[u]) {
            int v = e.to;
            int cost = e.cost;
            // 如果从起始点经过u到v的距离小于从起始点直接到v的距离，则更新距离
            if (dist[u] + cost < dist[v]) {
                dist[v] = dist[u] + cost;
                pq.push({dist[v], v});
            }
        }
    }

    return dist;
}

int main() {
    // 创建图
    int V = 5; // 顶点数
    Graph graph(V);

    // 添加边
    graph.addEdge(0, 1, 10);
    graph.addEdge(0, 2, 5);
    graph.addEdge(1, 2, 2);
    graph.addEdge(1, 3, 1);
    graph.addEdge(2, 1, 3);
    graph.addEdge(2, 3, 9);
    graph.addEdge(2, 4, 2);
    graph.addEdge(3, 4, 4);
    graph.addEdge(4, 0, 7);
    graph.addEdge(4, 3, 6);

    // 从顶点0开始计算最短路径
    vector<int> dist = dijkstra(graph, 0);

    // 输出结果
    for (int i = 0; i < V; ++i) {
        cout << "Distance from vertex 0 to vertex " << i << " is " << dist[i] << endl;
    }

    return 0;
}
