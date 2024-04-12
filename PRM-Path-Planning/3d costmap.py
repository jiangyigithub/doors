import numpy as np
import matplotlib.pyplot as plt

class GridMap:
    def __init__(self, map_size, resolution):
        self.map_size = map_size  # 地图尺寸 (x, y, theta)
        self.resolution = resolution  # 分辨率 (x_res, y_res, theta_res)
        self.grid_size = (int(map_size[0] / resolution[0]), int(map_size[1] / resolution[1]), int(map_size[2] / resolution[2]))  # 栅格地图尺寸 (x_size, y_size, theta_size)
        self.grid_map = np.zeros(self.grid_size, dtype=int)  # 初始化栅格地图

    def world_to_grid(self, x, y, theta):
        x_idx = int((x + self.map_size[0] / 2) / self.resolution[0])
        y_idx = int((y + self.map_size[1] / 2) / self.resolution[1])
        theta_idx = int(theta / self.resolution[2])
        return x_idx, y_idx, theta_idx

    def set_obstacle(self, x, y, theta):
        x_idx, y_idx, theta_idx = self.world_to_grid(x, y, theta)
        self.grid_map[x_idx, y_idx, theta_idx] = 1

    def clear_obstacle(self, x, y, theta):
        x_idx, y_idx, theta_idx = self.world_to_grid(x, y, theta)
        self.grid_map[x_idx, y_idx, theta_idx] = 0
    
    def visualize_map(self):
        plt.figure(figsize=(100, 100))
        plt.imshow(self.grid_map[:, :, 0], cmap='binary', origin='lower')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('2D Grid Map')
        plt.colorbar(label='Obstacle')
        plt.grid(True, which='both', linestyle='--', color='gray', linewidth=0.5)
        plt.show()

    def print_obstacle_coordinates(self):
        for theta_idx in range(self.grid_size[2]):
            for y_idx in range(self.grid_size[1]):
                for x_idx in range(self.grid_size[0]):
                    if self.grid_map[x_idx, y_idx, theta_idx] == 1:
                        print(f"Obstacle at grid coordinates: ({x_idx}, {y_idx}, {theta_idx})")

# 定义地图尺寸和分辨率
map_size = (100, 100, 360)  # (x, y, theta)
resolution = (10, 10, 10)  # (x_res, y_res, theta_res)

# 创建栅格地图对象
grid_map = GridMap(map_size, resolution)

# 在地图上设置障碍物
grid_map.set_obstacle(7.5, 8.5, 45)  # 在大地坐标 (10, 20, 45) 处设置障碍物

# 可视化2D栅格地图
grid_map.visualize_map()

# 打印障碍物的栅格坐标
grid_map.print_obstacle_coordinates()
