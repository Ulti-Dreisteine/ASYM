from scipy.signal import welch
import scipy.signal
import numpy as np
import sys
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.abspath(__file__), "../" * 2))
sys.path.insert(0, BASE_DIR)

from setting import plt
from mod.dir_file_op import load_json_file
from core.ARX_process import ARXProcess
from s1_sysident import load_samples, setup_subplot_axes

if __name__ == "__main__":
    # 加载数据
    t_samples, u_samples, y_samples, e_samples = load_samples()

    # 加载辨识所得参数
    opt_arx_params = load_json_file("runtime/s1_optimal_arx_params.json") 

    # ---- 逐通道计算误差界 ---------------------------------------------------------------------------

    Ns, Dy, Du = y_samples.shape[0], y_samples.shape[1], u_samples.shape[1]

    err_var_results = {}

    for i in range(Dy):
        for j in range(Du):
            channel_key = f"y{i+1}_u{j+1}"

            order = opt_arx_params[channel_key]["order"]
            R = opt_arx_params[channel_key]["R"]
            A = np.array(opt_arx_params[channel_key]["A"])
            B = np.array(opt_arx_params[channel_key]["B"])
            C = np.array(opt_arx_params[channel_key]["C"])

            arx_process = ARXProcess(A, B, C)

            # 计算 n/N
            n_div_N = order / Ns

            # 输入的功率谱，采用归一化数字角频率
            f_u, Phi_u = welch(u_samples[:, j])

            # 将频率转换为角频率 (0 到 π)
            omega_u = 2 * np.pi * f_u

            # 计算 1/A(q) 在不同角频率的幅值响应平方
            _, h = scipy.signal.freqz(b=[1], a=A.flatten(), worN=omega_u)
            h_squared = np.abs(h) ** 2
            Phi_v = R * h_squared

            # 计算误差界
            error_var = n_div_N * Phi_v / Phi_u
            
            err_var_results[channel_key] = {
                "omega": omega_u.tolist(),
                "Phi_u": Phi_u.tolist(),
                "Phi_v": Phi_v.tolist(),
                "error_var": error_var.tolist()
            }

    # 画图 - 所有通道放在一张大图里
    axes = setup_subplot_axes(Dy, Du)
    
    for i in range(Dy):
        for j in range(Du):
            channel_key = f"y{i+1}_u{j+1}"
            result = err_var_results[channel_key]

            omega = np.array(result["omega"])
            Phi_u = np.array(result["Phi_u"])
            Phi_v = np.array(result["Phi_v"])
            error_var = np.array(result["error_var"])
            
            ax = axes[i, j]
            
            # 转换为dB
            Phi_u_db = 10 * np.log10(Phi_u + 1e-10)
            Phi_v_db = 10 * np.log10(Phi_v + 1e-10)
            error_var_db = 10 * np.log10(error_var + 1e-10)
            
            # 在同一张图上绘制三条曲线
            ax.semilogx(omega, Phi_u_db, label=r"$\Phi_u$ (Input)", linewidth=1.5)
            ax.semilogx(omega, Phi_v_db, label=r"$\Phi_v$ (Noise)", linewidth=1.5)
            ax.semilogx(omega, error_var_db, label=r"$\varepsilon$ (error)", linewidth=1.5, color="red")
            
            ax.set_title(f"$y_{i+1} \\leftarrow u_{j+1}$", fontsize=12)
            ax.set_xlabel(r"$\omega$ (rad/sample)", fontsize=12)
            ax.set_ylabel("Magnitude (dB)", fontsize=12)
            ax.tick_params(labelsize=9)
            ax.grid(True)
            ax.legend(fontsize=10)
    
    plt.tight_layout()
    
    # 保存图像
    plt.savefig(f"{BASE_DIR}/img/s2_error_bound_estimation.png", dpi=450)
