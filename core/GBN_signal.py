import numpy as np


def gen_gbn(n_samples: int, amplitude: float = 1.0, ETsw: float = 5, Ts: float = 1.0, seed: int = None) -> np.ndarray:
    """
    生成广义二值噪声（GBN, Generalized Binary Noise）信号
    
    原理：
    GBN信号是一种在+amplitude和-amplitude之间切换的二值信号。
    在每个时间步长Ts，信号以概率P_switch = Ts/ETsw决定是否切换状态。
    这种随机切换机制使得信号的平均切换时间为ETsw，同时保持最短保持时间为Ts。
    GBN信号常用于系统辨识中的激励信号设计，因为它能够有效激励系统的动态特性。
    
    具体步骤：
    1. 初始化信号为随机的+amplitude或-amplitude
    2. 在每个时间步，以概率P_switch判断是否切换信号值
    3. 如果切换，则将当前值取反（+amplitude变为-amplitude，反之亦然）
    4. 重复步骤2-3直到生成n_samples个样本

    @Params:
        n_samples: 信号样本数
        amplitude: 信号幅度（默认1.0）
        ETsw: 平均切换时间（Expected Time of Switching，默认5）
        Ts: 采样时间/最短保持时间（默认1.0）
        seed: 随机种子，用于结果可重复（默认None）

    @Return:
        gbn_signal: 生成的GBN信号, shape = (n_samples,)
    """
    if seed is not None:
        np.random.seed(seed)

    P_switch = Ts / ETsw  # 切换概率
    
    gbn_signal = np.zeros((n_samples,))

    # 初始化为随机的+amplitude或-amplitude
    current_value = amplitude if np.random.rand() > 0.5 else -amplitude
    gbn_signal[0] = current_value

    for j in range(1, n_samples):
        # 以P_switch的概率切换信号值
        if np.random.rand() < P_switch:
            current_value = -current_value

        gbn_signal[j] = current_value
    
    return gbn_signal


def gen_gbn_zhu(n_samples: int, ETsw: int) -> np.ndarray:
    """
    生成 GBN 信号（严格按照给定 MATLAB 逻辑）

    @Params:
        n_samples: 样本数 N
        ETsw: 平均切换间隔，切换概率 psw = 1/ETsw

    @Return:
        1D numpy 数组，元素为 +1 或 -1，dtype=float32
    """
    assert n_samples >= 1 and ETsw >= 1, "n_samples 和 ETsw 必须为正整数"
    psw = 1.0 / ETsw
    R = np.random.rand(n_samples)

    # 初始符号由第一个随机数决定（与 MATLAB 中 if R(1)>0.5 一致）
    P_M0 = 1.0 if R[0] > 0.5 else -1.0

    # 在每个位置是否发生翻转（与 MATLAB 中 if (R(k) < psw) 一致）
    flips = (R < psw).astype(np.int32)

    # 累计翻转次数，若累积翻转次数为 m，则符号为 P_M0 * (-1)**m
    cumflips = np.cumsum(flips)
    u = P_M0 * ((-1) ** cumflips)

    return u.astype(np.float32, copy=False)