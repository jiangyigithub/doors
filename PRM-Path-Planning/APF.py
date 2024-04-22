
import numpy as np
import matplotlib.pyplot as plt
class APF:
    def __init__(self, start, goal, obstacles):
        self.start = start.astype(float)  # 起点
        self.goal = goal.astype(float)    # 终点
        self.obstacles = [obs.astype(float) for obs in obstacles]  # 障碍物
        self.alpha = 50.0  # 吸引力系数
        self.beta = 100.0  # 斥力系数
        self.epsilon = 1e-6  # 防止除零
        self.eta = 0.1  # 步长参数

    def attractive_force(self, q):
        """计算吸引力"""
        return self.alpha * (self.goal - q) / np.linalg.norm(self.goal - q)

    def repulsive_force(self, q):
        """计算斥力"""
        f_rep = np.zeros_like(q)
        for obstacle in self.obstacles:
            dist = np.linalg.norm(obstacle - q)
            if dist < self.epsilon:
                dist = self.epsilon
            f_rep += (self.beta / dist**2) * ((q - obstacle) / dist)
        return f_rep

    def total_force(self, q):
        """计算总力"""
        return self.attractive_force(q) + self.repulsive_force(q)

    def gradient_total_force(self, q):
        """计算总势能场的梯度"""
        epsilon = 1e-6
        gradient_x = (self.total_force(np.array([q[0] + epsilon, q[1]])) - self.total_force(np.array([q[0] - epsilon, q[1]]))) / (2 * epsilon)
        gradient_y = (self.total_force(np.array([q[0], q[1] + epsilon])) - self.total_force(np.array([q[0], q[1] - epsilon]))) / (2 * epsilon)
        return np.array([gradient_x, gradient_y])

    def move(self, q):
        """根据总势能场的梯度移动机器人"""
        return q + self.eta * self.gradient_total_force(q)

    def plan_path(self, max_iter=1000):
        """规划路径"""
        path = [self.start]
        for _ in range(max_iter):
            q = path[-1]
            if np.linalg.norm(q - self.goal) < 0.5:  # 到达目标
                path.append(self.goal)  # 将终点添加到路径中
                break
            q_new = self.move(q)  # 根据总势能场的梯度移动机器人
            path.append(q_new)
        return np.array(path)

# 定义起点、终点和障碍物
start = np.array([1, 1])
goal = np.array([10, 10])
obstacles = [np.array([5, 5]), np.array([7, 8]), np.array([3, 6])]

# 初始化APF
apf = APF(start, goal, obstacles)

# 计划路径
path = apf.plan_path()

# 可视化地图和路径
plt.figure()
plt.plot(start[0], start[1], 'go', label='Start')
plt.plot(goal[0], goal[1], 'ro', label='Goal')
for obstacle in obstacles:
    plt.plot(obstacle[0], obstacle[1], 'bx', label='Obstacle')
plt.plot(path[:, 0], path[:, 1], 'b-', label='Path')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Artificial Potential Field Path Planning')
plt.legend()
plt.grid(True)
plt.axis('equal')
plt.show()
