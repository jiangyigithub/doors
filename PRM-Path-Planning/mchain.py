import numpy as np

# 定义状态和转移概率矩阵
states = ["Sunny", "Rainy"]
transition_probs = np.array([
    [0.8, 0.2],  # 从晴天转移的概率
    [0.4, 0.6]   # 从雨天转移的概率
])

def simulate_weather(initial_state, transition_probs, num_days):
    state = initial_state
    weather_sequence = [state]
    
    for _ in range(num_days - 1):
        state_index = states.index(state)
        next_state = np.random.choice(states, p=transition_probs[state_index])
        weather_sequence.append(next_state)
        state = next_state
    
    return weather_sequence

# 初始状态为晴天
initial_state = "Sunny"
num_days = 10

weather_sequence = simulate_weather(initial_state, transition_probs, num_days)
print("Weather sequence over {} days:".format(num_days))
print(weather_sequence)
