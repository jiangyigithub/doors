import numpy as np
import matplotlib.pyplot as plt

class RRT:
    def __init__(self, start, goal, obstacles, step_size, max_iter):
        self.start = start
        self.goal = goal
        self.obstacles = obstacles
        self.step_size = step_size # 步长
        self.max_iter = max_iter
        self.tree = {tuple(start): None}
        self.iterations = 0  # 追踪迭代次数

    def generate_random_point(self):
        return np.random.uniform(low=0, high=10, size=2)  

    def nearest_neighbor(self, point):
        distances = [np.linalg.norm(np.array(point) - np.array(node)) for node in self.tree.keys()]
        min_index = np.argmin(distances)
        return list(self.tree.keys())[min_index]

    def new_point(self, nearest, random):
        direction = np.array(random) - np.array(nearest)
        distance = np.linalg.norm(direction)
        if distance > self.step_size:
            direction = direction / distance * self.step_size
        return tuple(np.array(nearest) + direction)
    
    def collision_free(self, nearest, new):
        for obstacle in self.obstacles:
            # 检查线段是否与障碍物相交
            if self.intersects(obstacle, nearest, new):
                return False
        return True

    def intersects(self, obstacle, nearest, new):
        # 提取障碍物坐标
        obstacle_min = obstacle[0]
        obstacle_max = obstacle[1]
        
        # 检查线段是否与障碍物的任何边相交
        x1, y1 = nearest
        x2, y2 = new
        
        # 检查是否与矩形的垂直边相交
        if (obstacle_min[0] <= x1 <= obstacle_max[0] or obstacle_min[0] <= x2 <= obstacle_max[0]) and \
        (min(y1, y2) <= obstacle_max[1] <= max(y1, y2) or min(y1, y2) <= obstacle_min[1] <= max(y1, y2)):
            return True
        
        # 检查是否与矩形的水平边相交
        if (obstacle_min[1] <= y1 <= obstacle_max[1] or obstacle_min[1] <= y2 <= obstacle_max[1]) and \
        (min(x1, x2) <= obstacle_max[0] <= max(x1, x2) or min(x1, x2) <= obstacle_min[0] <= max(x1, x2)):
            return True
        
        return False


    def build_tree(self):
        for _ in range(self.max_iter):
            self.iterations += 1  # 每次迭代增加计数
            random_point = self.generate_random_point()  # 步骤 1：生成一个随机点
            nearest_point = self.nearest_neighbor(random_point)  # 步骤 2：找到树中最近的点
            new_point = self.new_point(nearest_point, random_point)  # 步骤 3：生成一个新点
            if self.collision_free(nearest_point, new_point):  # 步骤 4：检查碰撞
                self.tree[new_point] = nearest_point  # 步骤 5：将新点添加到树中
                if np.linalg.norm(np.array(new_point) - np.array(self.goal)) < self.step_size:  # 步骤 6：检查是否到达目标
                    self.tree[tuple(self.goal)] = new_point
                    return True
        return False

    def find_path(self):
        if not self.build_tree():
            return None

        path = [tuple(self.goal)]
        current = tuple(self.goal)
        while current != tuple(self.start):
            current = self.tree[current]
            path.append(current)
        return path[::-1]

def visualize(rrt, obstacles, path=None):
    plt.figure(figsize=(8, 6))
    
    # 绘制障碍物
    for obstacle in obstacles:
        plt.plot([obstacle[0][0], obstacle[1][0], obstacle[1][0], obstacle[0][0], obstacle[0][0]], 
                 [obstacle[0][1], obstacle[0][1], obstacle[1][1], obstacle[1][1], obstacle[0][1]], 'k-')
    
    # 绘制RRT树
    for node, parent in rrt.tree.items():
        if parent:
            plt.plot([node[0], parent[0]], [node[1], parent[1]], 'b-', alpha=0.5)
    
    # 绘制起点和目标点
    plt.plot(rrt.start[0], rrt.start[1], 'go', label='Start')
    plt.plot(rrt.goal[0], rrt.goal[1], 'ro', label='Goal')
    
    # 绘制最终路径
    if path:
        for i in range(len(path) - 1):
            plt.plot([path[i][0], path[i+1][0]], [path[i][1], path[i+1][1]], 'g-', linewidth=2)
    
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('RRT Path Planning')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    start = (1, 1)
    goal = (10, 10)
    obstacles = [((2, 3), (4, 5)), ((6, 7), (8, 9))]  # 定义障碍物坐标
    rrt = RRT(start, goal, obstacles, step_size=0.5, max_iter=5000)

    path = rrt.find_path()
    if path:
        print("Found a path with", rrt.iterations, "iterations:", path)
        visualize(rrt, obstacles, path=path) 
    else:
        print("No path found after", rrt.iterations, "iterations.") 
