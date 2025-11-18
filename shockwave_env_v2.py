import os
import numpy as np
import torch
import random

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)


class Agent:
    def __init__(self, idx, pos):
        self.idx = idx
        self.pos = np.array(pos, dtype=float)
        self.velocity = np.zeros(2)
        self.local_intensity = 0.0
        self.last_messages = []


class ShockwaveEnvV2:
    def __init__(self,
                 n_agents=3,
                 area_size=1000.0,
                 wave_speed=340.0,   # 当作 v0
                 dt=0.015,
                 T=350,
                 alpha=0.08,         # 衰减系数
                 beta=40.0,          # 非线性扰动振幅
                 omega=2 * np.pi * 0.5,  # 非线性扰动频率 (0.5 Hz)
                 v_min=50.0):        # 速度下限，防止在场地内停滞
        """
        n_agents: agent数量
        area_size: 场景尺寸 (m)
        wave_speed: v0，初始波速 (m/s)
        dt: 每帧采样间隔 (s)，工业传感典型为15 ms
        T: 总帧数
        alpha: 衰减系数
        beta, omega: 非线性扰动幅度与角频率
        v_min: 速度下限，避免 v(t) 变成 0
        """
        self.n_agents = n_agents
        self.area_size = area_size
        self.dt = dt
        self.T = T

        # v(t) = v0 * exp(-alpha t) + beta * sin(omega t)
        self.v0 = wave_speed
        self.alpha = alpha
        self.beta = beta
        self.omega = omega
        self.v_min = v_min

        self.agents = []
        self._init_agents()



    def _init_agents(self):
        """随机初始化agent位置"""
        self.agents = []
        for i in range(self.n_agents):
            pos = np.random.uniform(0, self.area_size, size=2)
            self.agents.append(Agent(i, pos))

    def simulate_one_event(self, src_pos=None, base_intensity=1.0):

        """模拟一次声波冲击事件"""
        X_seq, mask_seq, conf_seq = [], [], []
        conf_real_seq = []
        if src_pos is None:
            src_pos = np.random.uniform(0, self.area_size, size=2)

        t_hit = np.zeros(self.n_agents)
        dist_mat = np.zeros((self.n_agents, self.n_agents))
        dist_src_all = np.zeros(self.n_agents)      # ⭐ 每个 agent 到源点的距离
        hit_idx = np.zeros(self.n_agents, dtype=int)  # ⭐ 每个 agent 的命中帧索引

        # === 观测噪声与传感器失效配置（全局假设） ===
        NOISE_LOW, NOISE_HIGH = 0.05, 0.10  # ±5% ~ ±10%
        FAIL_P = 0.30  # 独立失效概率 30%

        # ===== 预计算 v(t) 与波前半径 R(t) =====
        v_t = np.zeros(self.T, dtype=float)
        R_t = np.zeros(self.T, dtype=float)

        for t in range(self.T):
            t_now = t * self.dt
            v = self.v0 * np.exp(-self.alpha * t_now) + self.beta * np.sin(self.omega * t_now)
            # 保证速度不小于 v_min
            v = max(v, self.v_min)
            v_t[t] = v
            if t == 0:
                R_t[t] = v * self.dt
            else:
                R_t[t] = R_t[t - 1] + v * self.dt

        # ===== 根据 R(t) 反推每个 agent 的命中时间 t_hit 和命中帧 hit_idx =====
        for i, ai in enumerate(self.agents):
            dist = np.linalg.norm(ai.pos - src_pos)
            dist_src_all[i] = dist

            # 第一个 R_t >= dist 的时间就是命中时间
            indices = np.where(R_t >= dist)[0]
            if len(indices) > 0:
                hi = int(indices[0])
            else:
                hi = self.T - 1  # 理论上不该发生，有 v_min 保底

            hit_idx[i] = hi
            t_hit[i] = hi * self.dt

            for j, aj in enumerate(self.agents):
                dist_mat[i, j] = np.linalg.norm(ai.pos - aj.pos)

        for t in range(self.T):
            t_now = t * self.dt
            v_now = v_t[t]  # 当前帧波速
            R_now = R_t[t]  # ⭐ 当前帧波前半径
            features, mask, conf_list = [], [], []
            conf_real_list = []

            # === 工业通信模型：随机设备延迟 5–10帧 ===
            messages = []
            for i, ai in enumerate(self.agents):
                msg_i = []
                for j, aj in enumerate(self.agents):
                    if i == j:
                        continue
                    delay_steps = np.random.randint(5, 11)
                    if t - delay_steps >= 0 and len(aj.last_messages) >= delay_steps:
                        msg_i.append(aj.last_messages[-delay_steps][0])
                    else:
                        msg_i.append(0.0)

                messages.append(np.mean(msg_i) if len(msg_i) else 0.0)

            # === agent状态更新 ===
            for i, ai in enumerate(self.agents):
                # 源点到 agent 的几何距离
                dist_src = np.linalg.norm(ai.pos - src_pos)

                # ⭐ 距波前距离：>0 表示波还没到，=0 表示已被波扫过
                dist_to_front = max(dist_src - R_now, 0.0)

                # ⭐ 速度传感器测的是当前波速，而不是 agent 自身速度
                vel_mag = v_now

                # --- 加入观测噪声与独立失效 ---
                def noisy_obs_with_meta(val):
                    amp = np.random.uniform(NOISE_LOW, NOISE_HIGH)
                    eps = np.random.uniform(-amp, amp)
                    val_noisy = val * (1.0 + eps)
                    fail = (np.random.rand() < FAIL_P)
                    return (0.0 if fail else val_noisy), amp, eps, fail

                # ⭐ 现在距离观测基于 dist_to_front，速度观测基于 v_now
                dist_obs, amp_d, eps_d, fail_d = noisy_obs_with_meta(dist_to_front)
                vel_obs, amp_v, eps_v, fail_v = noisy_obs_with_meta(vel_mag)

                # --- 置信度标签计算（理想标签，用于Oracle比较）---
                if not (fail_d or fail_v):
                    # 观测相对方差（噪声幅度平方）
                    var_norm = 0.5 * (eps_d ** 2 + eps_v ** 2)
                    # 历史一致性（当前与前一帧差值的相对偏差）
                    if t > 0:
                        prev_d = X_seq[-1][i][3]  # 上一帧的 dist_obs（第4列）
                        prev_v = X_seq[-1][i][2]  # 上一帧的 vel_obs（第3列）
                        cons = 0.5 * (
                                abs(dist_obs - prev_d) / (abs(prev_d) + 1e-6)
                                + abs(vel_obs - prev_v) / (abs(prev_v) + 1e-6)
                        )
                    else:
                        cons = 0.0
                    conf_label = np.exp(-(var_norm + cons))
                else:
                    conf_label = 0.0

                # === NEW：现实可实现的 heuristic 置信度 ===
                # 只依赖于当前/上一帧观测，不用 eps/fail 等“作弊”信息
                if t > 0:
                    prev_d_obs = X_seq[-1][i][3]  # 上一帧 dist_obs（归一化前）
                    prev_v_obs = X_seq[-1][i][2]  # 上一帧 vel_obs
                    delta_d = abs(dist_obs - prev_d_obs) / (abs(prev_d_obs) + 1e-6)
                    delta_v = abs(vel_obs - prev_v_obs) / (abs(prev_v_obs) + 1e-6)
                    smooth_term = 0.5 * (delta_d + delta_v)
                else:
                    smooth_term = 0.0

                # 简单缺失度：两个观测都为 0 视作缺失/失效
                missing_term = 1.0 if (dist_obs == 0.0 and vel_obs == 0.0) else 0.0

                conf_real = np.exp(-(smooth_term + 0.5 * missing_term))
                # --- 强度：基于实时波速 v_now，波未到=0 ---
                intensity = base_intensity * (v_now / (self.v0 + 1e-6)) * (t_now >= t_hit[i])
                ai.local_intensity = intensity
                ai.last_messages.append((intensity,))
                if len(ai.last_messages) > 3000:
                    ai.last_messages.pop(0)

                # --- 特征与标签记录 ---
                feat = [ai.pos[0], ai.pos[1], vel_obs, dist_obs, intensity, messages[i]]
                features.append(feat)
                mask.append(1.0 if t_now >= t_hit[i] else 0.0)
                # 新增：理想置信度标签（同维度 [N]）
                conf_list.append(conf_label)
                conf_real_list.append(conf_real)

            X_seq.append(features)
            mask_seq.append(mask)
            conf_seq.append(conf_list)
            conf_real_seq.append(conf_real_list)

        X_seq = np.array(X_seq)
        mask_seq = np.array(mask_seq)
        conf_seq = np.array(conf_seq)
        conf_real_seq = np.array(conf_real_seq)  # ⭐ [T,N]

        #  归一化到 [0,1]
        X_min = X_seq.min(axis=(0, 1), keepdims=True)
        X_max = X_seq.max(axis=(0, 1), keepdims=True)
        scale = (X_max - X_min)
        scale[scale < 1e-8] = 1.0
        X_seq = (X_seq - X_min) / scale

        # ===== 新增：每个 agent 的“撞击瞬间强度”（基于接触速度） =====
        impact_speed = v_t[hit_idx]  # shape: [n_agents]
        impact_intensity = base_intensity * (impact_speed / (self.v0 + 1e-6))

        return X_seq, mask_seq, conf_seq, conf_real_seq, src_pos, t_hit, impact_intensity


    def generate_dataset(self, n_runs=200, out_dir="data/generated_2000"):
        os.makedirs(out_dir, exist_ok=True)
        n_train = int(0.8 * n_runs)
        n_val = int(0.1 * n_runs)
        n_test = n_runs - n_train - n_val

        for run_id in range(n_runs):
            self._init_agents()
            # 现在多返回 impact_intensity
            X_seq, mask_seq, conf_seq, conf_real_seq, src, t_hit, impact_intensity = self.simulate_one_event()
            X_seq = X_seq.reshape(1, *X_seq.shape)
            mask_seq = mask_seq.reshape(1, *mask_seq.shape)
            conf_seq = conf_seq.reshape(1, *conf_seq.shape)
            conf_real_seq = conf_real_seq.reshape(1, *conf_real_seq.shape)
            impact_intensity = impact_intensity.reshape(1, -1, 1)

            if run_id < n_train:
                split = "train"
            elif run_id < n_train + n_val:
                split = "val"
            else:
                split = "test"

            np.savez(os.path.join(out_dir, f"snapshot_task1_run{run_id:03d}_{split}.npz"),
                     X=X_seq,
                     wave_mask=mask_seq,
                     conf_label=conf_seq,  # 上界 oracle 用
                     conf_real=conf_real_seq,  # NEW: 现实 heuristic oracle 用
                     impact_intensity=impact_intensity,          # ⭐ 新增字段
                     positions=[a.pos for a in self.agents],
                     t_hit=t_hit,
                     src=src,
                     params={
                         "v0": self.v0,
                         "alpha": self.alpha,
                         "beta": self.beta,
                         "omega": self.omega,
                         "v_min": self.v_min,
                         "dt": self.dt,
                     },
                     run_id=run_id, split=split)

            print(f"[OK] Saved run {run_id:03d} ({split}) shape={X_seq.shape}")



if __name__ == "__main__":
    env = ShockwaveEnvV2(n_agents=3, T=350)
    env.generate_dataset(n_runs=2000)
