import numpy as np
import matplotlib.pyplot as plt

class APF:
    def __init__(self, start, goal, obstacles):
        self.start = start.astype(float)  # 显式转换为浮点数类型
        self.goal = goal.astype(float)  # 显式转换为浮点数类型
        self.obstacles = [obs.astype(float) for obs in obstacles]  # 显式转换为浮点数类型
        self.alpha = 50.0  # 吸引力系数
        self.beta = 100.0  # 斥力系数
        self.epsilon = 1e-6  # 防止除零

    def attractive_force(self, q):
        return self.alpha * (self.goal - q) / np.linalg.norm(self.goal - q)

    def repulsive_force(self, q):
        f_rep = np.zeros_like(q)
        for obstacle in self.obstacles:
            dist = np.linalg.norm(obstacle - q)
            if dist < self.epsilon:
                dist = self.epsilon
            f_rep += (self.beta / dist**2) * ((q - obstacle) / dist)
        return f_rep

    def total_force(self, q):
        return self.attractive_force(q) + self.repulsive_force(q)

    def move(self, q, eta):
        return q + eta * self.total_force(q)

    def plan_path(self, max_iter=1000, eta=0.1):
        path = [self.start]
        for _ in range(max_iter):
            q = path[-1]
            if np.linalg.norm(q - self.goal) < 0.5:  # 到达目标
                break
            q_new = self.move(q, eta)
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
