# -*- coding: utf-8 -*-
"""
Created on 2025/11/01 21:30:02

@File -> ASYM_SISO.py

@Author: luolei

@Email: dreisteine262@163.com

@Describe: SISO过程ASYM辨识
"""

from typing import Literal
from gekko import GEKKO
import numpy as np

from core.ARX_process import ARXProcess

if "plt" not in globals():
    import matplotlib.pyplot as plt
else:
    plt = globals()["plt"]


class SISOARXIdentifier:
    """SISO过程ARX模型辨识器"""

    def __init__(self, u: np.ndarray, y: np.ndarray):
        """
        初始化SISOARXIdentifier实例

        @Params:
            u: 输入信号, shape = (n_samples,)
            y: 输出信号, shape = (n_samples,)
        """
        self.u = u.flatten()
        self.y = y.flatten()

    def identify(self, na: int, nb: int, mode: Literal["meas", "model"] = "meas", objf: float = 100):
        """
        使用GEKKO进行ARX模型辨识

        @Params:
            na: A多项式阶数
            nb: B多项式阶数
            mode: 预测模式, "meas"表示使用测量值进行预测, "model"表示使用模型预测值进行预测
            objf: 目标函数权重

        @Return:
            dict: 包含辨识结果的字典, 包括A, B, C多项式系数
        """
        m = GEKKO(remote=False)
        m.options.RTOL = 1e-6
        m.options.OTOL = 1e-6

        _, p, _ = m.sysid(
            np.zeros(len(self.u)), self.u, self.y,
            na=na, nb=nb, diaglevel=0, pred=mode, objf=objf
        )

        A = np.array([1] + [-p["a"].flatten()[k] for k in range(na)]).reshape(1, 1, -1)
        B = np.array([0] + [p["b"].flatten()[k] for k in range(nb)]).reshape(1, 1, -1)
        C = np.array([1]).reshape(1, 1, -1)
        
        # 计算噪声方差
        arx_process = ARXProcess(A=A, B=B, C=C)
        y_pred = arx_process.one_step_predict(
            u_samples=self.u.reshape(-1, 1),
            y_samples=self.y.reshape(-1, 1)
        ).flatten()
        R = np.var(self.y - y_pred)

        return {"na": na, "nb": nb, "A": A, "B": B, "C": C, "R": R}

    def order_reduction_plot(self, benchmark_order: int, order_lst: list | np.ndarray):
        """
        SISO模型降阶分析

        @Params:
            benchmark_order: 作为基准的高阶模型阶数
            order_lst: 需要评估的模型阶数列表
        """
        # 辨识benchmark模型
        benchmark_params = self.identify(na=benchmark_order, nb=benchmark_order)
        arx_benchmark = ARXProcess(A=benchmark_params["A"], B=benchmark_params["B"], C=benchmark_params["C"])
        y_benchmark = arx_benchmark.one_step_predict(
            u_samples=self.u.reshape(-1, 1),
            y_samples=self.y.reshape(-1, 1)
        ).flatten()

        # 计算各阶数的AIC
        N = len(y_benchmark)
        aic_lst = []
        
        for order in order_lst:
            params = self.identify(na=order, nb=order)
            arx_model = ARXProcess(A=params["A"], B=params["B"], C=params["C"])
            y_pred = arx_model.one_step_predict(
                u_samples=self.u.reshape(-1, 1),
                y_samples=self.y.reshape(-1, 1)
            ).flatten()
            
            mse = np.mean((y_benchmark - y_pred) ** 2)
            aic_lst.append(N * np.log(mse) + 4 * order)

        # 绘图
        plt.plot(order_lst, aic_lst, "k-o", linewidth=1.5, markersize=6)
        plt.xlabel("Model Order", fontsize=12)
        plt.ylabel("AIC", fontsize=12)
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.tick_params(labelsize=12)
