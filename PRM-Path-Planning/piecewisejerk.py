import numpy as np
from scipy.optimize import minimize

# 定义优化目标函数
def objective(x, w_s, w_v, w_a, w_j, N):
    s = x[:N+1]
    v = x[N+1:2*(N+1)]
    a = x[2*(N+1):3*(N+1)]
    j = x[3*(N+1):]
    cost = w_s * np.sum(s**2) + w_v * np.sum(v**2) + w_a * np.sum(a**2) + w_j * np.sum(j**2)
    return cost

# 定义约束
def constraint_eq(x, s0, v0, a0, s_target, v_target, dt, N):
    s = x[:N+1]
    v = x[N+1:2*(N+1)]
    a = x[2*(N+1):3*(N+1)]
    j = x[3*(N+1):]

    constraints = []
    
    # 初始条件约束
    constraints.append(s[0] - s0)
    constraints.append(v[0] - v0)
    constraints.append(a[0] - a0)
    
    # 终端条件约束
    constraints.append(s[-1] - s_target)
    constraints.append(v[-1] - v_target)
    
    # 动力学约束
    for i in range(N):
        constraints.append(s[i+1] - s[i] - v[i] * dt - 0.5 * a[i] * dt**2)
        constraints.append(v[i+1] - v[i] - a[i] * dt)
        constraints.append(a[i+1] - a[i] - j[i] * dt)

    return np.array(constraints)

# 参数设置
N = 10  # 时间步数
dt = 0.1  # 时间步长
s0, v0, a0 = 0, 0, 0  # 初始状态
s_target, v_target = 10, 0  # 目标状态
w_s, w_v, w_a, w_j = 1, 1, 1, 1  # 权重系数

# 初始猜测
x0 = np.zeros(4 * (N+1))

# 约束条件
constraints = {'type': 'eq', 'fun': constraint_eq, 'args': (s0, v0, a0, s_target, v_target, dt, N)}

# 优化
result = minimize(objective, x0, args=(w_s, w_v, w_a, w_j, N), constraints=constraints, method='SLSQP')

# 输出结果
s_opt = result.x[:N+1]
v_opt = result.x[N+1:2*(N+1)]
a_opt = result.x[2*(N+1):3*(N+1)]
j_opt = result.x[3*(N+1):]

print("Optimal position:", s_opt)
print("Optimal velocity:", v_opt)
print("Optimal acceleration:", a_opt)
print("Optimal jerk:", j_opt)
