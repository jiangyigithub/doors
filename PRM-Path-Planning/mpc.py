import numpy as np
from cvxopt import matrix, solvers  # 用于求解凸优化问题的库
import time
import matplotlib.pyplot as plt
import copy
import logging
 
# 设置日志记录器
logger = logging.getLogger("mpc-smoother")
logger.setLevel(level=logging.INFO)
handler = logging.FileHandler("mpc.log")
handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(levelname)s- %(filename)s:%(lineno)d - [%(message)s]")
handler.setFormatter(formatter)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
console.setFormatter(formatter)
logger.addHandler(handler)  # 输出到文件
# logger.addHandler(console)  # 输出到屏幕
 
epsilon = 1e-6  # 浮点数比较精度
 
class MpcSmoother:
    def __init__(self, delta_t, n_pre=11, n_save=3):
        self.N_pre = n_pre  # 预测步长
        self.N_save = n_save  # 平滑保存步长
        assert self.N_save < self.N_pre  # 要求保存步长小于预测步长
        self.delta_t = delta_t  # 时间步长
 
        self.At = None  # 状态转移矩阵，维度为 (2*Np, 2)
        self.Bt = None  # 控制输入矩阵，维度为 (2*Np, Np)
        self.Qt = None  # 加权矩阵，维度为 (2*Np, 2*Np)
        self.Rt = None  # 控制输入加权矩阵，维度为 (Np, Np)
 
        self.P = None  # QP问题中的矩阵P，维度为 (Np, Np)
        self.G = None  # QP问题中的约束矩阵G，维度为 (2*Np, Np)
        self.ub = 3.2  # 控制输入上限
        self.lb = -3.2  # 控制输入下限
        self.h = None  # QP问题中的约束向量h，维度为 (2*Np, 1)，由lb和ub组成的列向量
 
        self.reset_matrix(delta_t)  # 初始化系统矩阵和QP问题参数
 
    def interpolate_by_time(self, t, tsv_profile):
        # 根据时间t对轨迹数据进行插值，得到对应的位置和速度
        def interpolate(x1, x2, x, y1, y2):
            return y2 - (x2 - x) * (y2 - y1) / (x2 - x1)
 
        # 找到最接近的两个数据点，进行线性插值
        min_index = 0
        for i in range(len(tsv_profile)):
            if tsv_profile[i][0] >= t:
                min_index = i
                break
 
        if min_index > 0:
            p1 = tsv_profile[min_index - 1]
            p2 = tsv_profile[min_index]
        else:
            p1 = tsv_profile[min_index]
            p2 = tsv_profile[min_index + 1]
       
        # 插值计算位置s和速度v
        s = interpolate(p1[0], p2[0], t, p1[1], p2[1])
        v = interpolate(p1[0], p2[0], t, p1[2], p2[2])
 
        return [t, s, v]
 
    def get_q(self, x0, Xr):
        # 计算二次规划问题中的q向量
        q = 2 * x0.T.dot(self.At.T).dot(self.Qt).dot(self.Bt) \
            - 2 * Xr.T.dot(self.Qt).dot(self.Bt)
        return matrix(q).T
 
    def reset_matrix(self, delta_t):
        # 重置系统矩阵和QP问题参数
        self.delta_t = delta_t
 
        # 系统动态矩阵和控制输入矩阵
        A = np.array([[1, delta_t], [0, 1]])
        B = np.array([[delta_t**2 / 2], [delta_t]])
 
        # 权重矩阵Q和R
        Q = np.array([[3, 0], [0, 1]])
        R = np.array([[3]])
 
        # 初始化系统矩阵和权重矩阵
        At = copy.deepcopy(A)
        Bt = copy.deepcopy(B)
        Qt = copy.deepcopy(Q)
        Rt = copy.deepcopy(R)
 
        A_k = np.array([[1, delta_t], [0, 1]])
        temp = np.array([[delta_t**2 / 2], [delta_t]])
 
        # 构建预测步长内的系统矩阵和权重矩阵
        for _ in range(2, self.N_pre + 1):
            Bt = np.hstack((Bt, np.zeros((Bt.shape[0], B.shape[1]))))
            temp = np.hstack((A_k.dot(B), temp))
            Bt = np.vstack((Bt, temp))
 
            A_k = np.dot(A_k, A)
            At = np.vstack((At, A_k))
 
            Qt = np.hstack((Qt, np.zeros((Qt.shape[0], Q.shape[0]))))
            q = np.hstack((np.zeros((Q.shape[0], Qt.shape[0])), Q))
            Qt = np.vstack((Qt, q))
 
            Rt = np.hstack((Rt, np.zeros((Rt.shape[0], R.shape[0]))))
            r = np.hstack((np.zeros((R.shape[0], Rt.shape[0])), R))
            Rt = np.vstack((Rt, r))
 
        # 构建QP问题中的矩阵P、G和约束向量h
        P = Bt.T.dot(Qt).dot(Bt) + Rt
        self.P = matrix(P)
        self.At = At
        self.Bt = Bt
        self.Qt = Qt
        self.Rt = Rt
        self.G = matrix(np.vstack((np.identity(self.N_pre), -np.identity(self.N_pre))))
        ub = np.array([self.ub] * self.N_pre).reshape(self.N_pre, 1)
        lb = np.array([self.lb] * self.N_pre).reshape(self.N_pre, 1)
        h = np.vstack((ub, -lb))
        self.h = matrix(h)
 
    def smooth(self, tsv_profile):
        t1 = time.time()
        N = int((tsv_profile[-1][0] - tsv_profile[0][0]) / self.delta_t)
 
        ref_t_arr = []
        ref_s_arr = []
        ref_v_arr = []
        for i in range(0, N + 1):
            t = i * self.delta_t
            _, s, v = self.interpolate_by_time(t, tsv_profile)
            ref_t_arr.append(t)
            ref_s_arr.append(s)
            ref_v_arr.append(v)
 
        i = 0
        tsv_smooth = [[ref_t_arr[i], ref_s_arr[i], ref_v_arr[i]]]
        i += 1
        while i + self.N_pre < len(ref_t_arr):
            x0 = np.array([tsv_smooth[-1][1], tsv_smooth[-1][2]]).reshape(2, 1)
            Xr = []
            for j in range(i, i + self.N_pre):
                Xr.append(ref_s_arr[j])
                Xr.append(ref_v_arr[j])
            Xr = np.array(Xr)
 
            q = self.get_q(x0, Xr)
            result = solvers.qp(P=self.P, q=q, G=self.G, h=self.h, kktsolver='ldl', options={'kktreg': 1e-9})
               
            U = result['x']
            Xt = self.At.dot(x0).reshape(2 * self.N_pre, 1) + self.Bt.dot(U)
 
            pre_t = tsv_smooth[-1][0]
            for k in range(self.N_save):
                t = self.delta_t * (k + 1) + pre_t
                s = Xt[2 * k]
                v = Xt[2 * k + 1]
                tsv_smooth.append([t, s[0], v[0]])
 
            i += self.N_save
 
        t2 = time.time()
        logger.info("mpc smooth used time: {:.6f}s".format(t2 - t1))
        return tsv_smooth
 
 
def test_mpc_smoother():
    tsv_profile = [[0, 0.0, 20.0], [1, 22.0, 20.0], [2, 40.0, 22.0],
                   [3, 60.0, 20.0], [4, 80.0, 20.0], [5, 101.0, 20.0],
                   [6, 120.0, 20.0]] # 需要追踪的曲线
 
    delta_t = 0.1
    mpc_smoother = MpcSmoother(delta_t)
    tsv_smooth = mpc_smoother.smooth(tsv_profile)
 
    plt.figure()
    ax_s = plt.gca()
    ax_s.set_ylabel("distance[m]")
    ax_s.set_xlabel("time[s]")
 
    plt.figure()
    ax_v = plt.gca()
    ax_v.set_ylabel("speed[m/s]")
    ax_v.set_xlabel("time[s]")
 
    t_profile = [tsv[0] for tsv in tsv_profile]
    s_profile = [tsv[1] for tsv in tsv_profile]
    v_profile = [tsv[2] for tsv in tsv_profile]
 
    t_smooth = [tsv[0] for tsv in tsv_smooth]
    s_smooth = [tsv[1] for tsv in tsv_smooth]
    v_smooth = [tsv[2] for tsv in tsv_smooth]
 
    ax_s.plot(t_profile, s_profile, label="s_profile")
    ax_s.plot(t_smooth, s_smooth, label="s_smooth")
    ax_v.plot(t_profile, v_profile, label="v_profile")
    ax_v.plot(t_smooth, v_smooth, label="v_smooth")
 
    ax_s.legend()
    ax_v.legend()
    plt.show()
    print("debug")
 
 
if __name__ == "__main__":
    test_mpc_smoother()