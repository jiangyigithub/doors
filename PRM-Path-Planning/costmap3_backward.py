import math
import numpy as np
import matplotlib.pyplot as plt
import heapq

# 定义节点类
class Node:
    def __init__(self, state, g_cost, h_cost, parent, depth):
        self.state = state
        self.g_cost = g_cost
        self.h_cost = h_cost
        self.parent = parent
        self.depth = depth

    # 定义节点比较方法，用于堆排序
    def __lt__(self, other):
        return (self.g_cost + self.h_cost) < (other.g_cost + other.h_cost)

# 定义车辆的运动模型
class VehicleModel:
    def __init__(self):
        self.length = 1  # 车辆长度
        self.max_steering_angle = math.radians(45)  # 最大转向角度
        self.width = 0.5  # 车辆宽度

    def get_successors(self, state, obstacles):
        successors = []

        # 定义可能的移动方向
        directions = [(1, 0), (-1, 0)]  # 分别表示向前、向后

        # 定义可能的转向角度
        angles = [0, self.max_steering_angle,-self.max_steering_angle]  # 分别表示直行、向左转、向右转
        # angles = [0, self.max_steering_angle, math.radians(30),-math.radians(30),-self.max_steering_angle]  # 分别表示直行、向左转、向右转

        # 当前状态的位置和方向
        x, y, yaw = state

        # 遍历所有可能的方向和转向角度
        for dx, dy in directions:
            for angle in angles:
                # 计算新状态的位置和方向
                new_x = x + dx * math.cos(yaw)
                new_y = y + dx * math.sin(yaw)
                new_yaw = yaw + angle

                # 判断新状态是否合法，并添加到后继状态列表中
                if not self.check_collision((new_x, new_y, new_yaw), obstacles):
                    successors.append(((new_x, new_y, new_yaw), angle))

        return successors

    def check_collision(self, state, obstacles):
        x, y, _ = state
        for obstacle in obstacles:
            ox, oy, _ = obstacle
            distance = math.sqrt((x - ox) ** 2 + (y - oy) ** 2)
            if distance < self.width:
                return True
        return False

# 定义 Hybrid A* 算法
def hybrid_a_star(start_state, goal_state, vehicle_model, costmap, obstacles, max_search_depth):
    open_list = []
    closed_set = set()

    start_node = Node(start_state, 0, costmap[int(start_state[0]), int(start_state[1])], None, 0)
    heapq.heappush(open_list, start_node)

    while open_list:
        current_node = heapq.heappop(open_list)  # 取出具有最小代价的节点
        current_state, g_cost, h_cost, parent, depth = current_node.state, current_node.g_cost, current_node.h_cost, current_node.parent, current_node.depth

        if heuristic(current_state, goal_state) < 0.01:  # 当当前状态接近目标状态时停止搜索
            path = []
            while current_node:
                path.append(current_node.state)
                current_node = current_node.parent
            path.reverse()
            return path

        closed_set.add(current_state)

        if depth > max_search_depth:
            return None  # 如果达到最大搜索深度

        successors = vehicle_model.get_successors(current_state, obstacles)
        for successor_state, action in successors:
            if successor_state in closed_set:
                continue
            g_cost_successor = g_cost + 1
            h_cost_successor = costmap[int(successor_state[0]), int(
                successor_state[1])]  # 修正索引
            new_node = Node(successor_state, g_cost_successor,
                            h_cost_successor, current_node, depth + 1)
            heapq.heappush(open_list, new_node)
    return None

# 定义启发式函数
def heuristic(state, goal_state):
    return math.sqrt((state[0] - goal_state[0]) ** 2 + (state[1] - goal_state[1]) ** 2)

# 生成栅格地图和启发代价
def generate_costmap(grid_size, goal_state):
    rows, cols = grid_size
    costmap = np.zeros((rows, cols))
    for i in range(rows):
        for j in range(cols):
            costmap[i, j] = math.sqrt(
                (i - goal_state[0]) ** 2 + (j - goal_state[1]) ** 2)
    return costmap

# 绘制路径
def plot_path(path, obstacles, grid_size, costmap):
    plt.figure(figsize=(8, 8))

    # 绘制 cost map
    plt.imshow(costmap, cmap='gray', origin='lower',
               extent=[0, grid_size[0], 0, grid_size[1]])

    for obstacle in obstacles:
        circle = plt.Circle(
            (obstacle[0], obstacle[1]), obstacle[2], color='red')
        plt.gca().add_patch(circle)

    x = [state[0] for state in path]
    y = [state[1] for state in path]
    plt.plot(x, y, color='blue', linewidth=2)
    plt.scatter(x, y, color='blue', s=30)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Hybrid A* Path')
    plt.axis('equal')
    plt.xticks(np.arange(0, grid_size[0] + 1, 1))
    plt.yticks(np.arange(0, grid_size[1] + 1, 1))
    plt.grid(True)
    plt.show()


# 测试
start_state = (0, 0, 0)
# start_state = (0, 0, math.pi / 4)
goal_state = (5, 5, math.pi / 4)
vehicle_model = VehicleModel()
obstacles = [(2, 2, 0.5), (3, 3, 0.5)]  # 障碍物位置和半径
grid_size = (10, 10)  # 栅格地图尺寸
max_search_depth = 12  # 最大搜索层数

# 生成栅格地图和启发代价
costmap = generate_costmap(grid_size, (goal_state[0], goal_state[1]))

# 执行 Hybrid A* 算法
path = hybrid_a_star(start_state, goal_state, vehicle_model,
                     costmap, obstacles, max_search_depth)
print("Hybrid A* 找到的路径：", path)

# 绘制路径
if path:
    plot_path(path, obstacles, grid_size, costmap)
