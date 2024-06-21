import numpy as np

# 定义网格大小和状态空间
grid_size = (4, 3)
states = [(i, j) for i in range(grid_size[0]) for j in range(grid_size[1])]
actions = ["U", "D", "L", "R"]
num_states = len(states)
num_actions = len(actions)

# 定义奖励函数
rewards = np.zeros(grid_size)
rewards[3, 2] = 1  # 目标位置的奖励
rewards[1, 1] = -1  # 障碍物的惩罚
rewards[1, 0] = rewards[1, 2] = rewards[2, 1] = rewards[3, 0] = -1  # 无效移动的惩罚

# 定义转移概率函数
def transition_prob(state, action):
    i, j = state
    if action == "U":
        return (max(i - 1, 0), j)
    elif action == "D":
        return (min(i + 1, grid_size[0] - 1), j)
    elif action == "L":
        return (i, max(j - 1, 0))
    elif action == "R":
        return (i, min(j + 1, grid_size[1] - 1))
    else:
        return state

# 初始化价值函数
V = np.zeros(grid_size)
gamma = 0.9  # 折扣因子
theta = 0.0001  # 价值迭代的收敛阈值

# 价值迭代算法
def value_iteration():
    while True:
        delta = 0
        for state in states:
            if state == (3, 2):
                continue  # 目标位置的价值不变
            i, j = state
            v = V[i, j]
            V[i, j] = max(rewards[i, j] + gamma * V[transition_prob(state, a)] for a in actions)
            delta = max(delta, abs(v - V[i, j]))
        if delta < theta:
            break

value_iteration()

# 策略提取
policy = np.zeros(grid_size, dtype=str)
for state in states:
    if state == (3, 2):
        policy[state] = 'G'
    else:
        policy[state] = max(actions, key=lambda a: V[transition_prob(state, a)])

print("Optimal Value Function:")
print(V)
print("Optimal Policy:")
print(policy)
