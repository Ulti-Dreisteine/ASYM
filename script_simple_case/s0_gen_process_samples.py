# -*- coding: utf-8 -*-
"""
Created on 2025/11/01 14:22:06

@File -> s0_生成一个ARX过程.py

@Author: luolei

@Email: dreisteine262@163.com

@Describe: 生成ARX过程
"""

import numpy as np
import sys
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.abspath(__file__), "../" * 2))
sys.path.insert(0, BASE_DIR)

from setting import plt
from core.ARX_process import ARXProcess
from core.GBN_signal import gen_gbn


if __name__ == "__main__":

    # ---- 初始化ARX参数 -----------------------------------------------------------------------------

    A = np.array([
        [[1, -1.5, 0.7], [0, 0, 0]],
        [[0, 0, 0], [1, -1.4, 0.65]]
    ])

    B = np.array([
        [[0, 1, 0.5], [0, 0, 1]],
        [[0, 0.75, 0.25], [0, 1, -0.3]]
    ])

    C = np.array([
        [[1, -1, 0.2], [0, 0, 0]],
        [[0, 0, 0], [1, -0.9, 0]]
    ])

    noise_var = np.array([[0.1, 0], [0, 0.1]])
    

    # ---- 创建ARX过程 ------------------------------------------------------------------------------

    self = ARXProcess(A=A, B=B, C=C)

    # ---- 计算阶跃响应 ------------------------------------------------------------------------------
    
    n_steps = 50
    t_response, y_responses = self.step_response(n_steps=n_steps, show_fig=True)

    # 保存图片
    if not os.path.exists(f"{BASE_DIR}/img"):
        os.makedirs(f"{BASE_DIR}/img")
    
    plt.savefig(f"{BASE_DIR}/img/s0_arx_step_response.png", dpi=450)

    # ---- 进行仿真模拟 ------------------------------------------------------------------------------
    
    n_samples = 10000
    
    # 输入为广义二值噪声GBN
    u_samples = np.zeros((n_samples, self.nu))
    
    for i in range(self.nu):
        u_samples[:, i] = gen_gbn(n_samples=n_samples, amplitude=1.0, ETsw=5, seed=2 + i)

    # 噪声为高斯白噪声
    e_samples = np.random.multivariate_normal(mean=np.zeros(self.ny), cov=noise_var, size=n_samples)

    t_samples, u_samples, y_samples, e_samples = self.simulate(u_samples=u_samples, e_samples=e_samples, show_fig=True)

    # 保存图片
    plt.savefig(f"{BASE_DIR}/img/s0_arx_simulation_results.png", dpi=450)

    # ---- 保存结果用于后续辨识分析 --------------------------------------------------------------------

    # 如果当前目录不存在runtime文件夹，则创建它
    if not os.path.exists("runtime"):
        os.makedirs("runtime")

    np.savez("runtime/s0_arx_process_data.npz",
             t_samples=t_samples,
             u_samples=u_samples,
             y_samples=y_samples,
             e_samples=e_samples)
    
