# -*- coding: utf-8 -*-
"""
Created on 2025/11/02 13:32:17

@File -> s1_sysident.py

@Author: luolei

@Email: dreisteine262@163.com

@Describe: 系统辨识
"""

import numpy as np
import sys
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.abspath(__file__), "../" * 2))
sys.path.insert(0, BASE_DIR)

from setting import plt
from mod.dir_file_op import save_json_file
from core.ARX_process import ARXProcess
from core.ASYM_SISO import SISOARXIdentifier


def load_samples():
    samples = np.load("runtime/s0_arx_process_data.npz", allow_pickle=True)
    return samples["t_samples"], samples["u_samples"], samples["y_samples"], samples["e_samples"]


def get_real_params():
    """获取实际过程参数"""
    A_real = np.array([[[1, -1.5, 0.7], [0, 0, 0]], [[0, 0, 0], [1, -1.4, 0.65]]])
    B_real = np.array([[[0, 1, 0.5], [0, 0, 1]], [[0, 0.75, 0.25], [0, 1, -0.3]]])
    C_real = np.array([[[1, -1, 0.2], [0, 0, 0]], [[0, 0, 0], [1, -0.9, 0]]])
    return A_real, B_real, C_real


def setup_subplot_axes(Dy, Du):
    """设置子图坐标轴"""
    _, axes = plt.subplots(Dy, Du, figsize=(5 * Du, 3 * Dy))
    axes = np.atleast_2d(axes)

    if Dy == 1:
        axes = axes.reshape(1, -1)
    elif Du == 1:
        axes = axes.reshape(-1, 1)
    else:
        pass

    return axes


def identify_all_channels(u_samples, y_samples, order):
    """对所有通道进行辨识"""
    Dy, Du = y_samples.shape[1], u_samples.shape[1]
    results = {}
    
    for i in range(Dy):
        for j in range(Du):
            identifier = SISOARXIdentifier(u=u_samples[:, j], y=y_samples[:, i])
            params = identifier.identify(na=order, nb=order)
            results[f"y{i+1}_u{j+1}"] = {k: v for k, v in params.items() if k in ["A", "B", "C"]}
    
    return results


def compare_step_responses(ident_results, u_samples, y_samples, title, labels_dict):
    """对比不同辨识结果的阶跃响应"""
    A_real, B_real, C_real = get_real_params()
    Dy, Du = y_samples.shape[1], u_samples.shape[1]
    axes = setup_subplot_axes(Dy, Du)

    for i in range(Dy):
        for j in range(Du):
            # 构建各过程
            processes = {}
            
            # 实际过程
            real_process = ARXProcess(
                A=A_real[i:i+1, i:i+1, :],
                B=B_real[i:i+1, j:j+1, :],
                C=C_real[i:i+1, i:i+1, :]
            )
            processes["Real"] = real_process
            
            # 辨识过程
            for key, results in ident_results.items():
                params = results[f"y{i+1}_u{j+1}"]
                label = labels_dict[key](i, j) if callable(labels_dict[key]) else labels_dict[key]
                processes[label] = ARXProcess(**params)
            
            # 绘制阶跃响应
            for label, process in processes.items():
                t_resp, y_resp = process.step_response(n_steps=50, show_fig=False)
                
                # 设置颜色
                if label == "Real":
                    color = "black"
                    linestyle = "-"
                elif "Benchmark" in label:
                    color = "blue"
                    linestyle = "--"
                else:  # Optimal
                    color = "red"
                    linestyle = "--"
                
                axes[i, j].plot(t_resp, y_resp[0, 0, :], linestyle=linestyle, 
                               color=color, linewidth=1.5, label=label)
            
            if i == Dy - 1:
                axes[i, j].set_xlabel("samples", fontsize=12)
            if j == 0:
                axes[i, j].set_ylabel(f"output $y_{i+1}$", fontsize=12)
            
            axes[i, j].set_title(f"$y_{i+1} \\leftarrow u_{j+1}$", fontsize=12)
            axes[i, j].legend(fontsize=10, frameon=True)
            axes[i, j].grid(True, linestyle="--", alpha=0.6)
            axes[i, j].tick_params(labelsize=12)

    plt.suptitle(title, fontsize=16)
    plt.tight_layout()


def compare_freq_responses(ident_results, u_samples, y_samples, title, labels_dict):
    """对比不同辨识结果的频率响应"""
    A_real, B_real, C_real = get_real_params()
    Dy, Du = y_samples.shape[1], u_samples.shape[1]
    axes = setup_subplot_axes(Dy, Du)

    for i in range(Dy):
        for j in range(Du):
            # 构建各过程
            processes = {}
            
            # 实际过程
            real_process = ARXProcess(
                A=A_real[i:i+1, i:i+1, :],
                B=B_real[i:i+1, j:j+1, :],
                C=C_real[i:i+1, i:i+1, :]
            )
            processes["Real"] = real_process
            
            # 辨识过程
            for key, results in ident_results.items():
                params = results[f"y{i+1}_u{j+1}"]
                label = labels_dict[key](i, j) if callable(labels_dict[key]) else labels_dict[key]
                processes[label] = ARXProcess(**params)
            
            # 绘制频率响应
            for label, process in processes.items():
                omega, mag = process.freq_response(show_fig=False)
                
                # 设置颜色
                if label == "Real":
                    color = "black"
                    linestyle = "-"
                elif "Benchmark" in label:
                    color = "blue"
                    linestyle = "--"
                else:  # Optimal
                    color = "red"
                    linestyle = "--"
                
                axes[i, j].semilogx(omega, mag[0, 0, :], linestyle=linestyle, color=color, linewidth=1.5, label=label)
            
            if i == Dy - 1:
                axes[i, j].set_xlabel("$\\omega$ (rad/sample)", fontsize=12)
            if j == 0:
                axes[i, j].set_ylabel("Magnitude (dB)", fontsize=12)
            
            axes[i, j].set_title(f"$y_{i+1} \\leftarrow u_{j+1}$", fontsize=12)
            axes[i, j].legend(fontsize=10, frameon=True)
            axes[i, j].grid(True, linestyle="--", alpha=0.6)
            axes[i, j].tick_params(labelsize=12)

    plt.suptitle(title, fontsize=16)
    plt.tight_layout()

if __name__ == "__main__":
    t_samples, u_samples, y_samples, e_samples = load_samples()
    Dy, Du = y_samples.shape[1], u_samples.shape[1]

    # ---- 辨识高阶模型 ------------------------------------------------------------------------------

    print("辨识高阶benchmark模型...")

    benchmark_order = 20
    benchmark_results = identify_all_channels(u_samples, y_samples, benchmark_order)

    compare_step_responses(
        {"benchmark": benchmark_results},
        u_samples, y_samples,
        "Comparison: Real vs Benchmark",
        {"benchmark": f"Benchmark (order={benchmark_order})"}
    )

    # 保存图片
    plt.savefig(f"{BASE_DIR}/img/s1_sysident_comparison_benchmark.png", dpi=450)

    # ---- 模型降阶分析 ------------------------------------------------------------------------------

    print("模型降阶分析...")

    order_lst = list(range(1, benchmark_order))
    axes = setup_subplot_axes(Dy, Du)
    
    for i in range(Dy):
        for j in range(Du):
            plt.sca(axes[i, j])
            
            identifier = SISOARXIdentifier(u=u_samples[:, j], y=y_samples[:, i])
            identifier.order_reduction_plot(benchmark_order=benchmark_order, order_lst=order_lst)
            
            axes[i, j].set_title(f"$y_{i+1} \\leftarrow u_{j+1}$", fontsize=12)
            axes[i, j].set_title(f"$y_{i+1} \\leftarrow u_{j+1}$", fontsize=12)
            if i == Dy - 1:
                axes[i, j].set_xlabel("Model Order", fontsize=12)
            if j == 0:
                axes[i, j].set_ylabel("AIC", fontsize=12)

    plt.suptitle("Model Order Selection via AIC", fontsize=16)
    plt.tight_layout()

    # 保存图片
    plt.savefig(f"{BASE_DIR}/img/s1_sysident_model_order_selection.png", dpi=450)

    # ---- 使用最佳阶数进行最终辨识 --------------------------------------------------------------------

    best_orders = {"y1_u1": 4, "y1_u2": 3, "y2_u1": 4, "y2_u2": 4}

    print("使用最佳阶数进行最终辨识...")
    
    opt_ident_results = {}
    opt_ident_results_serializable = {}
    
    for i in range(Dy):
        for j in range(Du):
            key = f"y{i+1}_u{j+1}"
            order = best_orders[key]
            
            identifier = SISOARXIdentifier(u=u_samples[:, j], y=y_samples[:, i])
            params = identifier.identify(na=order, nb=order)
            
            # 提取ABC参数
            opt_ident_results[key] = {k: params[k] for k in ["A", "B", "C"] if k in params}
            
            # 转换为可序列化格式,并加入order
            opt_ident_results_serializable[key] = {
                "order": order,
                "R": params["R"],
                **{
                    k: v.tolist() if isinstance(v, np.ndarray) else v 
                    for k, v in opt_ident_results[key].items()
                }
            }

    save_json_file(opt_ident_results_serializable, "runtime/s1_optimal_arx_params.json")

    # ---- 对比最优阶数、benchmark和实际过程 -----------------------------------------------------------

    compare_step_responses(
        {"benchmark": benchmark_results, "optimal": opt_ident_results}, u_samples, y_samples,
        "Step Response Comparison: Real vs Benchmark vs Optimal Order",
        {
            "benchmark": f"Benchmark (order={benchmark_order})",
            "optimal": lambda i, j: f"Optimal (order={best_orders[f'y{i+1}_u{j+1}']})"
        }
    )

    # 保存图片
    plt.savefig(f"{BASE_DIR}/img/s1_sysident_comparison_optimal.png", dpi=450)

    # ---- 对比频率响应 ------------------------------------------------------------------------------

    print("对比频率响应...")

    compare_freq_responses(
        {"benchmark": benchmark_results, "optimal": opt_ident_results}, u_samples, y_samples,
        "Frequency Response Comparison: Real vs Benchmark vs Optimal Order",
        {
            "benchmark": f"Benchmark (order={benchmark_order})",
            "optimal": lambda i, j: f"Optimal (order={best_orders[f'y{i+1}_u{j+1}']})"
        }
    )

    # 保存图片
    plt.savefig(f"{BASE_DIR}/img/s1_sysident_freq_comparison_optimal.png", dpi=450)