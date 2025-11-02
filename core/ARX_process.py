# -*- coding: utf-8 -*-
"""
Created on 2025/11/01 17:50:53

@File -> ARX_process.py

@Author: luolei

@Email: dreisteine262@163.com

@Describe: ARX过程
"""

from scipy.signal import welch
import scipy.signal
import numpy as np
import sys
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.abspath(__file__), "../" * 2))
sys.path.insert(0, BASE_DIR)

from setting import plt


class ARXProcess(object):
    """ARX过程类"""

    def __init__(self, A: np.ndarray, B: np.ndarray, C: np.ndarray):
        """
        初始化ARX过程：

        A(q)y(t) = B(q)u(t) + C(q)e(t)
        
        @Params:
            A: 多项式A矩阵，shape = (ny, ny, na)
            B: 多项式B矩阵, shape = (ny, nu, nb)
            C: 噪声模型C矩阵, shape = (ny, ny, nc)

        @Notes:
            ARX模型中C应该为单位矩阵
        """
        self.A = np.array(A)                   # 多项式A矩阵，shape = (ny, ny, na)
        self.B = np.array(B)                   # 多项式B矩阵, shape = (ny, nu, nb)
        self.C = np.array(C)                   # 噪声模型C矩阵, shape = (ny, ny, nc)

        self.ny = A.shape[0]  # 输出变量个数
        self.nu = B.shape[1]  # 输入变量个数
        self.na = A.shape[2]  # 输出的自回归阶数
        self.nb = B.shape[2]  # 输入的阶数
        self.nc = C.shape[2]  # 噪声的阶数

    def step_response(self, n_steps: int, show_fig: bool = True) -> tuple:
        """
        计算ARX过程的阶跃响应

        @Params:
            n_steps: 响应步数(不包括预运行阶段)
            show_fig: 是否显示图形

        @Return:
            t_response: 时间向量, shape = (n_steps,)
            y_responses: 阶跃响应输出, shape = (nu, ny, n_steps)
        """

        max_order = max(self.na, self.nb, self.nc)
        n_warmup = 10 * max_order  # 预运行步数为最大阶数的10倍
        n_total = n_warmup + n_steps  # 总步数
        
        y_responses = np.zeros((self.nu, self.ny, n_steps))

        # 对每个输入通道分别计算阶跃响应
        for input_idx in range(self.nu):

            # 生成单通道阶跃输入(预运行阶段为0,之后阶跃到1)
            u_response = np.zeros((n_total, self.nu))
            u_response[n_warmup:, input_idx] = 1.0
            y_response = np.zeros((n_total, self.ny))

            # 计算阶跃响应(包括预运行阶段)
            for t in range(max_order, n_total):
                for i in range(self.ny):
                    # 计算AR部分
                    ar_sum = 0.0
                    for j in range(self.ny):
                        for k in range(1, self.na):
                            ar_sum -= self.A[i, j, k] * y_response[t - k, j]

                    # 计算X部分
                    x_sum = 0.0
                    for j in range(self.nu):
                        for k in range(1, self.nb):
                            x_sum += self.B[i, j, k] * u_response[t - k, j]

                    # 输出当前时刻的值(阶跃响应中噪声为零)
                    y_response[t, i] = ar_sum + x_sum
            
            # 只保留阶跃后的n_steps个样本
            y_responses[input_idx, :, :] = y_response[n_warmup:, :].T
        
        # 时间向量从0开始,长度为n_steps
        t_response = np.arange(n_steps)

        if show_fig:
            # 为每对(u_i, y_j)创建单独的子图
            _, axes = plt.subplots(self.ny, self.nu, figsize=(3.5 * self.nu, 2 * self.ny))
            
            if self.nu == 1 or self.ny == 1:
                axes = np.array(axes).reshape(self.ny, self.nu)
            
            for i in range(self.nu):
                for j in range(self.ny):
                    axes[j, i].plot(t_response, y_responses[i, j, :], "k", linewidth=1.5)
                    axes[j, i].set_title(f"$u_{i+1}$ → $y_{j+1}$", fontsize=10)

                    if j == self.ny - 1:
                        axes[j, i].set_xlabel("samples")
                    if i == 0:
                        axes[j, i].set_ylabel("output")

                    axes[j, i].grid(True, alpha=0.3)

            plt.tight_layout()

        return t_response, y_responses

    def freq_response(self, worN: int = 512, show_fig: bool = True) -> tuple:
        """
        计算ARX过程的频率响应

        @Params:
            worN: 频率点数
            show_fig: 是否显示图形
            
        @Return:
            omega: 角频率向量, shape = (worN,)
            H: 频率响应矩阵, shape = (ny, nu, worN)
        """
        omega = np.linspace(0, np.pi, worN)  # 角频率从0到π
        H = np.zeros((self.ny, self.nu, worN), dtype=float)

        for j in range(self.ny):
            for i in range(self.nu):
                # 计算频率响应 H_ij(omega) = B_ij(e^(j*omega)) / A_ii(e^(j*omega))
                b = self.B[j, i, :].flatten()
                a = self.A[j, j, :].flatten()
                _, h_ij = scipy.signal.freqz(b=b, a=a, worN=omega, fs=2*np.pi)
                H[j, i, :] = np.real(h_ij)

        if show_fig:
            # 为每对(u_i, y_j)创建单独的子图,布局与step_response一致
            _, axes = plt.subplots(self.ny, self.nu, figsize=(3.5 * self.nu, 2 * self.ny))
            
            if self.nu == 1 or self.ny == 1:
                axes = np.array(axes).reshape(self.ny, self.nu)
            
            for i in range(self.nu):
                for j in range(self.ny):
                    axes[j, i].plot(omega, 20 * np.log10(np.abs(H[j, i, :])), "k", linewidth=1.5)
                    axes[j, i].set_title(f"$u_{i+1}$ → $y_{j+1}$", fontsize=10)

                    if j == self.ny - 1:
                        axes[j, i].set_xlabel("$\\omega$ (rad/sample)")
                    if i == 0:
                        axes[j, i].set_ylabel("Magnitude (dB)")

                    axes[j, i].grid(True, alpha=0.3)

            plt.tight_layout()

        return omega, H

    def simulate(self, u_samples: np.ndarray, e_samples: np.ndarray, show_fig: bool = True, cutoff: int = 500) -> tuple:
        """
        进行ARX过程仿真

        @Params:
            u_samples: 输入样本, shape = (n_samples, nu)
            e_samples: 噪声样本, shape = (n_samples, ny)
            show_fig: 是否显示图形

        @Return:
            t_samples: 时间向量, shape = (n_samples,)
            u_samples: 输入样本, shape = (n_samples, nu)
            y_samples: 输出样本, shape = (n_samples, ny)
            e_samples: 噪声样本, shape = (n_samples, ny)
        """

        n_samples = u_samples.shape[0]
        assert u_samples.shape == (n_samples, self.nu), "输入样本形状不匹配!"
        assert e_samples.shape == (n_samples, self.ny), "噪声样本形状不匹配!"

        # 计算输出样本
        y_samples = np.zeros((n_samples, self.ny))

        for t in range(max(self.na, self.nb, self.nc), n_samples):
            for i in range(self.ny):
                # 计算AR部分
                ar_sum = 0.0
                for j in range(self.ny):
                    for k in range(1, self.na):
                        ar_sum -= self.A[i, j, k] * y_samples[t - k, j]

                # 计算X部分
                x_sum = 0.0
                for j in range(self.nu):
                    for k in range(1, self.nb):
                        x_sum += self.B[i, j, k] * u_samples[t - k, j]

                # 计算C部分（噪声滤波）
                c_sum = 0.0
                for j in range(self.ny):
                    for k in range(1, self.nc):
                        c_sum += self.C[i, j, k] * e_samples[t - k, j]

                # 输出当前时刻的值
                y_samples[t, i] = ar_sum + x_sum + c_sum + e_samples[t, i]

        t_samples = np.arange(n_samples)

        # 画图
        if show_fig:
            _, axes = plt.subplots(3, 2, figsize=(2 * 3, 1.5 * 3))

            axes[0, 0].plot(t_samples[:cutoff], u_samples[:cutoff, 0], "k", linewidth=1.0)
            axes[0, 0].set_ylabel("$u_1$")
            axes[0, 0].grid(True, alpha=0.3)

            axes[0, 1].plot(t_samples[:cutoff], u_samples[:cutoff, 1], "k", linewidth=1.0)
            axes[0, 1].set_ylabel("$u_2$")
            axes[0, 1].grid(True, alpha=0.3)

            axes[1, 0].plot(t_samples[:cutoff], y_samples[:cutoff, 0], "k", linewidth=1.0)
            axes[1, 0].set_ylabel("$y_1$")
            axes[1, 0].grid(True, alpha=0.3)

            axes[1, 1].plot(t_samples[:cutoff], y_samples[:cutoff, 1], "k", linewidth=1.0)
            axes[1, 1].set_ylabel("$y_2$")
            axes[1, 1].grid(True, alpha=0.3)

            axes[2, 0].plot(t_samples[:cutoff], e_samples[:cutoff, 0], "k", linewidth=1.0)
            axes[2, 0].set_ylabel("$e_1$")
            axes[2, 0].set_xlabel("samples")
            axes[2, 0].grid(True, alpha=0.3)

            axes[2, 1].plot(t_samples[:cutoff], e_samples[:cutoff, 1], "k", linewidth=1.0)
            axes[2, 1].set_ylabel("$e_2$")
            axes[2, 1].set_xlabel("samples")
            axes[2, 1].grid(True, alpha=0.3)

            plt.tight_layout()

        return t_samples, u_samples, y_samples, e_samples
    
    def one_step_predict(self, u_samples: np.ndarray, y_samples: np.ndarray) -> np.ndarray:
        """
        进行ARX过程一步预测

        @Params:
            u_samples: 输入样本, shape = (n_samples, nu)
            y_samples: 输出样本, shape = (n_samples, ny)

        @Return:
            y_pred_samples: 预测输出样本, shape = (n_samples, ny)
        """

        n_samples = u_samples.shape[0]
        assert u_samples.shape == (n_samples, self.nu), "输入样本形状不匹配!"
        assert y_samples.shape == (n_samples, self.ny), "输出样本形状不匹配!"

        # 计算预测输出样本
        y_pred_samples = np.zeros((n_samples, self.ny))

        for t in range(max(self.na, self.nb), n_samples):
            for i in range(self.ny):
                # 计算AR部分
                ar_sum = 0.0
                for j in range(self.ny):
                    for k in range(1, self.na):
                        ar_sum -= self.A[i, j, k] * y_samples[t - k, j]

                # 计算X部分
                x_sum = 0.0
                for j in range(self.nu):
                    for k in range(1, self.nb):
                        x_sum += self.B[i, j, k] * u_samples[t - k, j]

                # 输出当前时刻的预测值
                y_pred_samples[t, i] = ar_sum + x_sum

        return y_pred_samples