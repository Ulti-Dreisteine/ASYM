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


def gen_gbn_signal(n_samples: int, n_signals: int, amplitude: float = 1.0, ETsw: float = 5, Ts: float = 1.0, seed: int = None) -> np.ndarray:
    """
    生成广义二值噪声（GBN, Generalized Binary Noise）信号
    注意：生成的各个信号之间是相互独立的

    @Params:
        n_samples: 信号样本数
        n_signals: 信号个数
        amplitude: 信号幅度
        ETsw: 平均切换时间
        Ts: 最短保持时间
        seed: 随机种子

    @Return:
        gbn_signal: 生成的GBN信号, shape = (n_samples, n_signals)
    """
    if seed is not None:
        np.random.seed(seed)

    p_switch = Ts / ETsw  # 切换概率
    
    gbn_signal = np.zeros((n_samples, n_signals))
    
    for i in range(n_signals):
        # 初始化为随机的+amplitude或-amplitude
        current_value = amplitude if np.random.rand() > 0.5 else -amplitude
        gbn_signal[0, i] = current_value
        
        for j in range(1, n_samples):
            # 以p_switch的概率切换信号值
            if np.random.rand() < p_switch:
                current_value = -current_value
                
            gbn_signal[j, i] = current_value
    
    return gbn_signal


def gen_prbs_signal(n_samples: int, n_signals: int, amplitude: float = 1.0, register_length: int = 7, seed: int = None) -> np.ndarray:
    """
    生成伪随机二进制序列（PRBS, Pseudo-Random Binary Sequence）信号
    注意：生成的各个信号之间是相互独立的

    @Params:
        n_samples: 信号样本数
        n_signals: 信号个数
        amplitude: 信号幅度
        register_length: 移位寄存器长度，决定序列周期为2^register_length - 1
        seed: 随机种子

    @Return:
        prbs_signal: 生成的PRBS信号, shape = (n_samples, n_signals)
    """
    if seed is not None:
        np.random.seed(seed)

    # PRBS反馈多项式抽头位置（基于最大长度序列）
    taps = {
        3: [3, 2],
        4: [4, 3],
        5: [5, 3],
        6: [6, 5],
        7: [7, 6],
        8: [8, 6, 5, 4],
        9: [9, 5],
        10: [10, 7],
    }
    
    if register_length not in taps:
        raise ValueError(f"register_length must be in {list(taps.keys())}")
    
    prbs_signal = np.zeros((n_samples, n_signals))
    
    for i in range(n_signals):
        # 初始化移位寄存器（非全零）
        register = np.random.randint(0, 2, size=register_length)
        if np.sum(register) == 0:
            register[0] = 1
        
        for j in range(n_samples):
            # 当前输出为寄存器最后一位
            output = register[-1]
            prbs_signal[j, i] = amplitude if output == 1 else -amplitude
            
            # 计算反馈位（XOR操作）
            feedback = 0
            for tap in taps[register_length]:
                feedback ^= register[tap - 1]
            
            # 移位并插入反馈位
            register = np.roll(register, 1)
            register[0] = feedback
    
    return prbs_signal


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
    u_samples = gen_gbn_signal(n_samples=n_samples, n_signals=self.nu, amplitude=1.0, ETsw=5, seed=2)

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
    
