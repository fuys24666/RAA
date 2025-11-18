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
    def __init__(self, x, y):
        self.pos = np.array([x, y], dtype=np.float32)
        self.last_messages = []


class ShockwaveEnvV1:
    """
    统一物理内核版 V1：
    - 物理传播模型与 V2 完全一致（v(t), R(t), t_hit, impact_intensity）
    - 用 SCENARIO_CONFIG 控制噪音 / 失效 / 延迟等干扰强度
    - 输出字段和 V2 一致：X / wave_mask / conf_label / conf_real / impact_intensity / positions ...
    """

    SCENARIO_CONFIG = {
        "clean": {
            "noise_low": 0,
            "noise_high": 0,
            "fail_p": 0,
            "msg_delay_min": 0,
            "msg_delay_max": 0,
        },
        "high_delay": {
            "noise_low": 0.05,
            "noise_high": 0.10,
            "fail_p": 0.30,
            "msg_delay_min": 15,
            "msg_delay_max": 30,
        },
        "high_fail": {
            "noise_low": 0.05,
            "noise_high": 0.10,
            "fail_p": 0.60,
            "msg_delay_min": 5,
            "msg_delay_max": 10,
        },
        "high_noise": {
            "noise_low": 0.20,
            "noise_high": 0.40,
            "fail_p": 0.30,
            "msg_delay_min": 5,
            "msg_delay_max": 10,
        },
        "combo": {
            "noise_low": 0.20,
            "noise_high": 0.40,
            "fail_p": 0.60,
            "msg_delay_min": 15,
            "msg_delay_max": 30,
        },
    }

    def __init__(self,
                 n_agents=3,
                 area_size=1000.0,
                 wave_speed=340.0,
                 dt=0.015,
                 T=350,
                 alpha=0.08,
                 beta=40.0,
                 omega=2 * np.pi * 0.5,
                 v_min=50.0,
                 scenario="clean"):
        self.n_agents = n_agents
        self.area_size = area_size
        self.dt = dt
        self.T = T

        # 统一物理内核：v(t) = v0 * exp(-alpha * t) + beta * sin(omega * t)
        self.v0 = float(wave_speed)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.omega = float(omega)
        self.v_min = float(v_min)

        # 为了兼容旧代码，保留 wave_speed 字段，但不再用于物理计算
        self.wave_speed = self.v0

        if scenario not in self.SCENARIO_CONFIG:
            raise ValueError(f"Unknown scenario: {scenario}")
        self.scenario = scenario

        self._init_agents()

    def _init_agents(self):
        self.agents = []
        for _ in range(self.n_agents):
            x = np.random.uniform(0, self.area_size)
            y = np.random.uniform(0, self.area_size)
            self.agents.append(Agent(x, y))

    def simulate_one_event(self, src_pos=None, base_intensity=1.0):
        cfg = self.SCENARIO_CONFIG[self.scenario]
        NOISE_LOW = cfg["noise_low"]
        NOISE_HIGH = cfg["noise_high"]
        FAIL_P = cfg["fail_p"]
        MSG_DELAY_MIN = cfg["msg_delay_min"]
        MSG_DELAY_MAX = cfg["msg_delay_max"]

        if src_pos is None:
            src_pos = np.random.uniform(0, self.area_size, size=2).astype(np.float32)

        # 预计算每个 agent 到源的距离
        dist_src_all = np.zeros(self.n_agents, dtype=np.float32)
        t_hit = np.zeros(self.n_agents, dtype=np.float32)
        hit_idx = np.zeros(self.n_agents, dtype=int)
        for i, a in enumerate(self.agents):
            d = np.linalg.norm(a.pos - src_pos)
            dist_src_all[i] = d

        # === 与 V2 统一的连续时间波速模型 v(t) 与波前 R(t) ===
        t_grid = np.arange(self.T, dtype=np.float32) * self.dt  # [T]
        v_t = self.v0 * np.exp(-self.alpha * t_grid) + self.beta * np.sin(self.omega * t_grid)
        v_t = np.maximum(v_t, self.v_min)  # 避免速度衰减到 0

        R_t = np.cumsum(v_t * self.dt)  # [T] 累积传播距离

        # 根据 R_t 与距离确定每个 agent 的命中时间（与 V2 一致）
        for i in range(self.n_agents):
            d = dist_src_all[i]
            hit_indices = np.where(R_t >= d)[0]
            if len(hit_indices) > 0:
                hit_idx[i] = int(hit_indices[0])
                t_hit[i] = t_grid[hit_idx[i]]
            else:
                hit_idx[i] = self.T - 1
                t_hit[i] = t_grid[-1]

        # 冲击瞬间强度：基于命中时刻的波速 v(t_hit)
        impact_speed = v_t[hit_idx]  # [N]
        impact_intensity = (base_intensity * (impact_speed / (self.v0 + 1e-6))).astype(np.float32)

        # 为通信延迟做准备
        for a in self.agents:
            a.last_messages = []

        X_seq, mask_seq, conf_seq, conf_real_seq = [], [], [], []

        for t in range(self.T):
            t_now = t * self.dt
            R_now = R_t[t]

            features, mask, conf_list, conf_real_list = [], [], [], []

            # 先生成带延迟的 messages（场景差异体现在 MSG_DELAY 范围）
            messages = []
            for i, ai in enumerate(self.agents):
                msg_i = []
                for j, aj in enumerate(self.agents):
                    if i == j:
                        continue
                    delay_steps = np.random.randint(MSG_DELAY_MIN, MSG_DELAY_MAX + 1)
                    if t - delay_steps >= 0 and len(aj.last_messages) > delay_steps:
                        msg_i.append(aj.last_messages[-delay_steps][0])
                    else:
                        msg_i.append(0.0)
                messages.append(np.mean(msg_i) if msg_i else 0.0)

            for i, ai in enumerate(self.agents):
                dist_src = dist_src_all[i]
                dist_to_front = max(dist_src - R_now, 0.0)
                v_now = v_t[t]
                vel_mag = v_now

                # 真实强度（物理场，与 V2 一致：基于实时波速）
                intensity = base_intensity * (v_now / (self.v0 + 1e-6)) if t_now >= t_hit[i] else 0.0

                # 噪声 + 失效（场景差异体现在 NOISE / FAIL_P）
                def noisy_obs_with_meta(val):
                    amp = np.random.uniform(NOISE_LOW, NOISE_HIGH)
                    eps = np.random.uniform(-amp, amp)
                    val_noisy = val * (1.0 + eps)
                    fail = (np.random.rand() < FAIL_P)
                    return (0.0 if fail else val_noisy), amp, eps, fail

                dist_obs, amp_d, eps_d, fail_d = noisy_obs_with_meta(dist_to_front)
                vel_obs, amp_v, eps_v, fail_v = noisy_obs_with_meta(vel_mag)

                # 上界 oracle 置信度（用 eps / fail）
                if not (fail_d or fail_v):
                    var_norm = 0.5 * (eps_d ** 2 + eps_v ** 2)
                    if t > 0:
                        prev_d = X_seq[-1][i][3]
                        prev_v = X_seq[-1][i][2]
                        cons = 0.5 * (
                            abs(dist_obs - prev_d) / (abs(prev_d) + 1e-6) +
                            abs(vel_obs - prev_v) / (abs(prev_v) + 1e-6)
                        )
                    else:
                        cons = 0.0
                    conf_label = np.exp(-(var_norm + cons))
                else:
                    conf_label = 0.0

                # 现实 heuristic 置信度（只用观测 & 历史；clean 场景下 conf_real ≈ conf_label）
                if t > 0:
                    prev_d_obs = X_seq[-1][i][3]
                    prev_v_obs = X_seq[-1][i][2]
                    delta_d = abs(dist_obs - prev_d_obs) / (abs(prev_d_obs) + 1e-6)
                    delta_v = abs(vel_obs - prev_v_obs) / (abs(prev_v_obs) + 1e-6)
                    smooth_term = 0.5 * (delta_d + delta_v)
                else:
                    smooth_term = 0.0

                missing_term = 1.0 if (dist_obs == 0.0 and vel_obs == 0.0) else 0.0
                conf_real = np.exp(-(smooth_term + 0.5 * missing_term))

                feat = [ai.pos[0], ai.pos[1], vel_obs, dist_obs, intensity, messages[i]]
                features.append(feat)
                mask.append(1.0 if t_now >= t_hit[i] else 0.0)
                conf_list.append(conf_label)
                conf_real_list.append(conf_real)

                ai.last_messages.append((intensity,))

            X_seq.append(features)
            mask_seq.append(mask)
            conf_seq.append(conf_list)
            conf_real_seq.append(conf_real_list)

        X_seq = np.array(X_seq, dtype=np.float32)          # [T,N,6]
        mask_seq = np.array(mask_seq, dtype=np.float32)    # [T,N]
        conf_seq = np.array(conf_seq, dtype=np.float32)    # [T,N]
        conf_real_seq = np.array(conf_real_seq, dtype=np.float32)  # [T,N]

        # 归一化到 [0,1]（按通道）
        X_min = X_seq.min(axis=(0, 1), keepdims=True)
        X_max = X_seq.max(axis=(0, 1), keepdims=True)
        X_range = np.where(X_max > X_min, X_max - X_min, 1.0)
        X_seq = (X_seq - X_min) / X_range

        # reshape 成 [1,T,N,F]，方便后面 DataLoader 堆叠
        X_seq = X_seq.reshape(1, *X_seq.shape)                 # [1,T,N,6]
        mask_seq = mask_seq.reshape(1, *mask_seq.shape)        # [1,T,N]
        conf_seq = conf_seq.reshape(1, *conf_seq.shape)        # [1,T,N]
        conf_real_seq = conf_real_seq.reshape(1, *conf_real_seq.shape)  # [1,T,N]
        impact_intensity = impact_intensity.reshape(1, -1, 1)  # [1,N,1]

        # positions：训练脚本期待的是 [N,2]
        positions = np.stack([a.pos for a in self.agents], axis=0).astype(np.float32)  # [N,2]

        return X_seq, mask_seq, conf_seq, conf_real_seq, src_pos, t_hit, impact_intensity, positions

    def generate_dataset(self, n_runs=200, out_dir="data/generated_v1"):
        os.makedirs(out_dir, exist_ok=True)
        n_train = int(0.8 * n_runs)
        n_val = int(0.1 * n_runs)
        n_test = n_runs - n_train - n_val

        for run_id in range(n_runs):
            self._init_agents()
            (X_seq,
             mask_seq,
             conf_seq,
             conf_real_seq,
             src,
             t_hit,
             impact_intensity,
             positions) = self.simulate_one_event()

            if run_id < n_train:
                split = "train"
            elif run_id < n_train + n_val:
                split = "val"
            else:
                split = "test"

            # 字段 & 形状与 V2 完全一致
            np.savez(
                os.path.join(out_dir, f"snapshot_task1_run{run_id:03d}_{split}.npz"),
                X=X_seq,                         # [1,T,N,6]
                wave_mask=mask_seq,              # [1,T,N]
                conf_label=conf_seq,             # [1,T,N]
                conf_real=conf_real_seq,         # [1,T,N]
                impact_intensity=impact_intensity,   # [1,N,1]
                positions=positions,             # [N,2]
                t_hit=t_hit,
                src=src,
                params={
                    "v0": self.v0,
                    "alpha": self.alpha,
                    "beta": self.beta,
                    "omega": self.omega,
                    "v_min": self.v_min,
                    "dt": self.dt,
                    "T": self.T,
                    "scenario": self.scenario,
                    # 兼容旧字段命名
                    "wave_speed": self.v0,
                },
                run_id=run_id,
                split=split,
            )

            print(f"[OK][{self.scenario}] Saved run {run_id:03d} ({split}) shape={X_seq.shape}")


def generate_all_scenarios_v1(n_runs_per_scenario=200, base_out_dir="data/generated_v1"):
    scenarios = ["clean", "high_delay", "high_fail", "high_noise", "combo"]
    for sc in scenarios:
        print(f"\n[INFO] Generating V1 dataset for scenario = {sc}")
        env = ShockwaveEnvV1(n_agents=3, T=350, scenario=sc)
        out_dir = os.path.join(base_out_dir, sc)
        env.generate_dataset(n_runs=n_runs_per_scenario, out_dir=out_dir)


if __name__ == "__main__":
    generate_all_scenarios_v1(n_runs_per_scenario=200)

