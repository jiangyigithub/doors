import numpy as np
import matplotlib.pyplot as plt

class ParkingAPF:
    def __init__(self, start, parking_spot, obstacles):
        self.start = start.astype(float)  # 起始位置
        self.parking_spot = parking_spot.astype(float)  # 停车位位置
        self.obstacles = [obs.astype(float) for obs in obstacles]  # 障碍物位置
        self.alpha = 5.0  # 吸引力系数
        self.beta = 10.0  # 斥力系数
        self.epsilon = 1e-6  # 防止除零
        self.eta = 0.1  # 步长参数

    def attractive_potential(self, q):
        return 0.5 * self.alpha * np.linalg.norm(self.parking_spot - q)**2

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

    def park(self, max_iter=1000):
        path = [self.start]
        for _ in range(max_iter):
            q = path[-1]
            if np.linalg.norm(q - self.parking_spot) < 0.5:  # 到达停车位
                break
            q_new = self.move(q)
            path.append(q_new)
        return np.array(path)

# 定义起点、停车位和障碍物
start = np.array([1, 1])
parking_spot = np.array([10, 5])
obstacles = [np.array([5, 3]), np.array([7, 7]), np.array([3, 6])]

# 初始化停车场APF
parking_apf = ParkingAPF(start, parking_spot, obstacles)

# 规划停车路径
path = parking_apf.park()

# 可视化
plt.figure()
plt.plot(start[0], start[1], 'go', label='Start')
plt.plot(parking_spot[0], parking_spot[1], 'ro', label='Parking Spot')
for obstacle in obstacles:
    plt.plot(obstacle[0], obstacle[1], 'bx', label='Obstacle')
plt.plot(path[:, 0], path[:, 1], 'b-', label='Path')
plt.plot(path[-1, 0], path[-1, 1], 'bo')  # 终点也标记为圆圈
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Parking with Artificial Potential Field')
plt.legend()
plt.grid(True)
plt.axis('equal')
plt.show()
