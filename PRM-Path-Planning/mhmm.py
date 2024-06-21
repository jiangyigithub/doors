import numpy as np

# 定义隐藏状态和观测状态
hidden_states = ['Rainy', 'Sunny']
observable_states = ['Walk', 'Shop', 'Clean']

# 初始概率分布
initial_prob = np.array([0.6, 0.4])  # 初始处于雨天和晴天的概率

# 状态转移概率矩阵
transition_probs = np.array([
    [0.7, 0.3],  # Rainy -> Rainy, Rainy -> Sunny
    [0.4, 0.6]   # Sunny -> Rainy, Sunny -> Sunny
])

# 观测概率矩阵
emission_probs = np.array([
    [0.1, 0.4, 0.5],  # Rainy: Walk, Shop, Clean
    [0.6, 0.3, 0.1]   # Sunny: Walk, Shop, Clean
])

# 观测序列
observations = ['Walk', 'Shop', 'Clean']

def forward_algorithm(obs_seq):
    T = len(obs_seq)  # 观测序列的长度
    N = len(hidden_states)  # 隐藏状态的数量
    
    # 初始化前向概率矩阵
    alpha = np.zeros((T, N))
    alpha[0, :] = initial_prob * emission_probs[:, observable_states.index(obs_seq[0])]
    
    # 递推计算前向概率
    for t in range(1, T):
        for j in range(N):
            alpha[t, j] = np.sum(alpha[t-1, :] * transition_probs[:, j]) * emission_probs[j, observable_states.index(obs_seq[t])]
    
    # 返回观测序列的概率，即最终时刻所有状态的前向概率之和
    return np.sum(alpha[T-1, :])

# 计算给定观测序列的概率
obs_probability = forward_algorithm(observations)
print(f"The probability of observing sequence {observations} is: {obs_probability}")
