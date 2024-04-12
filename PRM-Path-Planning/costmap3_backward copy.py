import math
import numpy as np
import matplotlib.pyplot as plt
import heapq

# Define the Node class
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

# Define the VehicleModel class
class VehicleModel:
    def __init__(self):
        self.length = 1  # Vehicle length
        self.max_steering_angle = math.radians(45)  # Maximum steering angle
        self.width = 0.5  # Vehicle width

    def get_successors(self, state, obstacles):
        successors = []
        directions = [(1, 0), (-1, 0)]  # Forward and backward directions
        angles = [0, self.max_steering_angle, -self.max_steering_angle]  # Straight, left, right

        x, y, yaw = state

        for dx, dy in directions:
            for angle in angles:
                new_x = x + dx * math.cos(yaw)
                new_y = y + dx * math.sin(yaw)
                new_yaw = yaw + angle

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

# Define the Hybrid A* algorithm
def hybrid_a_star(start_state, goal_state, vehicle_model, costmap, obstacles, max_search_depth):
    open_list = []
    closed_set = set()
    explored_nodes = []  # Store explored nodes

    start_node = Node(start_state, 0, costmap[int(start_state[0]), int(start_state[1])], None, 0)
    heapq.heappush(open_list, start_node)

    while open_list:
        current_node = heapq.heappop(open_list)
        current_state, g_cost, h_cost, parent, depth = current_node.state, current_node.g_cost, current_node.h_cost, current_node.parent, current_node.depth

        closed_set.add(current_state)
        explored_nodes.append(current_node)  # Add the current node to explored nodes

        if depth > max_search_depth:
            return None, explored_nodes

        if is_close_enough(current_state, goal_state):
            path = []
            while current_node:
                path.append(current_node.state)
                current_node = current_node.parent
            path.reverse()
            return path, explored_nodes

        successors = vehicle_model.get_successors(current_state, obstacles)
        for successor_state, action in successors:
            if successor_state in closed_set:
                continue
            g_cost_successor = g_cost + 1
            h_cost_successor = costmap[int(successor_state[0]), int(successor_state[1])]
            new_node = Node(successor_state, g_cost_successor, h_cost_successor, current_node, depth + 1)
            heapq.heappush(open_list, new_node)

    return None, explored_nodes

# Define the heuristic function
def heuristic(state, goal_state):
    position_difference = math.sqrt((state[0] - goal_state[0]) ** 2 + (state[1] - goal_state[1]) ** 2)
    yaw_difference = abs(state[2] - goal_state[2])
    yaw_difference = min(yaw_difference, 2 * math.pi - yaw_difference)
    return math.sqrt(position_difference ** 2 + yaw_difference ** 2)

# Define a function to check if the current state is close enough to the goal state
def is_close_enough(current_state, goal_state):
    position_difference = math.sqrt((current_state[0] - goal_state[0]) ** 2 + (current_state[1] - goal_state[1]) ** 2)
    yaw_difference = abs(current_state[2] - goal_state[2])
    yaw_difference = min(yaw_difference, 2 * math.pi - yaw_difference)
    return position_difference < 0.01 and yaw_difference < math.radians(1)

# Generate costmap and heuristic cost
def generate_costmap(grid_size, goal_state):
    rows, cols = grid_size
    costmap = np.zeros((rows, cols))
    for i in range(rows):
        for j in range(cols):
            costmap[i, j] = math.sqrt((i - goal_state[0]) ** 2 + (j - goal_state[1]) ** 2)
    return costmap

# Plot the explored nodes as blue lines
def plot_explored_nodes(explored_nodes, obstacles, grid_size):
    plt.figure(figsize=(8, 8))
    plt.imshow(np.zeros(grid_size), cmap='gray', origin='lower', extent=[0, grid_size[0], 0, grid_size[1]])
    for node in explored_nodes:
        if node.parent:
            x_values = [node.parent.state[0], node.state[0]]
            y_values = [node.parent.state[1], node.state[1]]
            plt.plot(x_values, y_values, color='blue', linewidth=1)
    for obstacle in obstacles:
        circle = plt.Circle((obstacle[0], obstacle[1]), obstacle[2], color='red')
        plt.gca().add_patch(circle)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Explored Nodes')
    plt.axis('equal')
    plt.grid(True)
    plt.show()

# Test
start_state = (0, 0, 0)
goal_state = (5, 5, math.pi / 4)
vehicle_model = VehicleModel()
obstacles = [(2, 2, 0.5), (3, 3, 0.5)]
grid_size = (10, 10)
max_search_depth = 12

costmap = generate_costmap(grid_size, (goal_state[0], goal_state[1]))

path, explored_nodes = hybrid_a_star(start_state, goal_state, vehicle_model, costmap, obstacles, max_search_depth)
print("Hybrid A* found path:", path)

if path:
    plot_explored_nodes(explored_nodes, obstacles, grid_size)
