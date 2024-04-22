import numpy as np
import matplotlib.pyplot as plt

class APF:
    def __init__(self, start, goal, obstacles):
        self.start = start.astype(float)  # 显式转换为浮点数类型
        self.goal = goal.astype(float)  # 显式转换为浮点数类型
        self.obstacles = [obs.astype(float) for obs in obstacles]  # 显式转换为浮点数类型
        self.alpha = 50.0  # 吸引力系数
        self.beta = 10.0  # 斥力系数
        self.epsilon = 1e-6  # 防止除零
        self.eta = 0.01  # 步长参数

    def attractive_potential(self, q):
        return 0.5 * self.alpha * np.linalg.norm(self.goal - q)**2

    def repulsive_potential(self, q):
        potential = 0.0
        for obstacle in self.obstacles:
            dist = np.linalg.norm(obstacle - q)
            if dist < self.epsilon:
                dist = self.epsilon
            potential += 0.5 * self.beta * (1 / dist**2)
        return potential

    def total_potential(self, q):
        return self.attractive_potential(q) + self.repulsive_potential(q)

    def gradient(self, q):
        delta = 1e-5
        dx = (self.total_potential(q + np.array([delta, 0])) - self.total_potential(q - np.array([delta, 0]))) / (2 * delta)
        dy = (self.total_potential(q + np.array([0, delta])) - self.total_potential(q - np.array([0, delta]))) / (2 * delta)
        return np.array([dx, dy])

    def move(self, q):
        return q + self.eta * self.gradient(q)

    def plan_path(self, max_iter=1000):
        path = [self.start]
        for _ in range(max_iter):
            q = path[-1]
            if np.linalg.norm(q - self.goal) < 0.5:  # 到达目标
                break
            q_new = self.move(q)
            path.append(q_new)
        return np.array(path)

# 定义起点、终点和障碍物
start = np.array([2, 2])
goal = np.array([8, 8])
obstacles = [np.array([4, 4]), np.array([6, 6]), np.array([3, 7])]

# 初始化APF
apf = APF(start, goal, obstacles)

# 计划路径
path = apf.plan_path()

# 可视化
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

