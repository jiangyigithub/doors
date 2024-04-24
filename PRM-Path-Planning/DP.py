import matplotlib.pyplot as plt
import numpy as np

def min_cost_path(grid):
    rows = len(grid)
    cols = len(grid[0])

    # 初始化DP数组和路径数组
    dp = [[0] * cols for _ in range(rows)]
    path = [[''] * cols for _ in range(rows)]

    # 计算第一行和第一列的累积代价以及路径
    dp[0][0] = grid[0][0]
    path[0][0] = str((0, 0))
    for i in range(1, rows):
        dp[i][0] = dp[i-1][0] + grid[i][0]
        path[i][0] = path[i-1][0] + ' -> ' + str((i, 0))
    for j in range(1, cols):
        dp[0][j] = dp[0][j-1] + grid[0][j]
        path[0][j] = path[0][j-1] + ' -> ' + str((0, j))

    # 计算其余格子的累积代价以及路径
    for i in range(1, rows):
        for j in range(1, cols):
            if dp[i-1][j] < dp[i][j-1]:
                dp[i][j] = dp[i-1][j] + grid[i][j]
                path[i][j] = path[i-1][j] + ' -> ' + str((i, j))
            else:
                dp[i][j] = dp[i][j-1] + grid[i][j]
                path[i][j] = path[i][j-1] + ' -> ' + str((i, j))

    # 返回最小代价和路径
    return dp[rows-1][cols-1], path[rows-1][cols-1]

def visualize_grid_with_path(grid, min_cost_path):
    rows = len(grid)
    cols = len(grid[0])

    # 创建网格
    fig, ax = plt.subplots()
    ax.imshow(grid, cmap='viridis', interpolation='nearest')

    # 绘制路径
    path_points = [eval(point) for point in min_cost_path.split(' -> ')]
    path_x = [point[1] for point in path_points]
    path_y = [point[0] for point in path_points]
    ax.plot(path_x, path_y, color='red', linewidth=3)

    # 显示代价值
    for i in range(rows):
        for j in range(cols):
            ax.text(j, i, str(grid[i][j]), ha='center', va='center', color='white')

    # 显示网格
    plt.title('Minimum Cost Path')
    plt.xlabel('Column')
    plt.ylabel('Row')
    plt.show()

# 示例
grid = [
    [1, 3, 1, 5],
    [1, 5, 1, 7],
    [4, 2, 1, 3],
    [5, 2, 8, 1]
]
min_cost, min_cost_path = min_cost_path(grid)
print("最小代价路径的总代价为:", min_cost)
print("最小代价路径为:", min_cost_path)
visualize_grid_with_path(grid, min_cost_path)

