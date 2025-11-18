import torch
import torch.nn as nn
import torch.nn.functional as F

class ConfEstimator(nn.Module):
    def __init__(self, ema_alpha=0.2):
        super().__init__()
        self.ema_alpha = ema_alpha
        self._a = nn.Parameter(torch.tensor(1.0))
        self._b = nn.Parameter(torch.tensor(1.0))
        self.softplus = nn.Softplus()

    def forward(self, obs_pair):  # obs_pair: [B,T,N,2] = (vel, dist)
        B, T, N, _ = obs_pair.shape
        vel, dist = obs_pair[..., 0], obs_pair[..., 1]
        ema_v, ema_d = vel[:, 0].clone(), dist[:, 0].clone()
        var_v = torch.zeros_like(ema_v); var_d = torch.zeros_like(ema_d)
        a = self.softplus(self._a); b = self.softplus(self._b); alpha = self.ema_alpha
        c_list = []
        for t in range(T):
            v_t, d_t = vel[:, t], dist[:, t]
            if t > 0:
                ema_v = alpha * v_t + (1 - alpha) * ema_v
                ema_d = alpha * d_t + (1 - alpha) * ema_d
                var_v = alpha * (v_t - ema_v).pow(2) + (1 - alpha) * var_v
                var_d = alpha * (d_t - ema_d).pow(2) + (1 - alpha) * var_d
            var_norm = var_v / (ema_v.abs() + 1e-6).pow(2) + var_d / (ema_d.abs() + 1e-6).pow(2)
            ema_dev = 0.5 * ((v_t - ema_v).abs()/(ema_v.abs()+1e-6) + (d_t - ema_d).abs()/(ema_d.abs()+1e-6))
            c_t = torch.exp(-(a * var_norm + b * ema_dev)).unsqueeze(-1)  # [B,N,1]
            c_list.append(c_t)
        return torch.stack(c_list, dim=1).clamp(1e-6, 1.0)  # [B,T,N,1]

class RAAGATLayer_self(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.1, alpha=0.2):
        super().__init__()
        self.W = nn.Linear(in_dim, out_dim, bias=False)
        self.a = nn.Linear(out_dim * 2 + 2, 1, bias=False)  # (h_i, h_j, c_j, d_ij)
        self.leakyrelu = nn.LeakyReLU(alpha)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h, conf, dist_mat, adj):
        """
        h: [B, N, F]
        conf: [B, N, 1]
        dist_mat: [N, N]
        adj: [N, N]
        """
        Wh = self.W(h)  # [B, N, out_dim]
        B, N, _ = Wh.size()

        Wh_i = Wh.unsqueeze(2).repeat(1, 1, N, 1)
        Wh_j = Wh.unsqueeze(1).repeat(1, N, 1, 1)
        c_j = conf.unsqueeze(1).repeat(1, N, 1, 1)
        d_ij = dist_mat.unsqueeze(0).unsqueeze(-1).repeat(B, 1, 1, 1)

        concat = torch.cat([Wh_i, Wh_j, c_j, d_ij], dim=-1)
        e = self.leakyrelu(self.a(concat)).squeeze(-1)

        e = e.masked_fill(adj == 0, float('-inf'))
        α = torch.softmax(e, dim=-1)
        α = self.dropout(α)
        out = torch.bmm(α, Wh)
        return out, α


class RAASTGNN_self(nn.Module):
    def __init__(self, in_dim=6, gcn_dim=64, gru_dim=64, dropout=0.2, num_heads=2):
        super().__init__()
        self.gat = RAAGATLayer_self(in_dim, gcn_dim, dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(
                d_model=gcn_dim, nhead=num_heads, dim_feedforward=128
            )
        self.temporal_transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.fc_I = nn.Linear(gcn_dim, 1)
        self.fc_mask = nn.Linear(gcn_dim, 1)
        self.dropout = nn.Dropout(dropout)
        # ★ 新增：内部置信度估计器 + 调试缓存
        self.conf_estimator = ConfEstimator(ema_alpha=0.2)
        self.last_alpha = None
        self.last_c_hat = None

    def forward(self, X, A, dist_mat):
        """
        X: [B, T, N, F]
        A: [N, N]
        dist_mat: [N, N]
        """
        B, T, N, F = X.size()

        # === 用 (vel, dist) 自估置信度 c_hat ：[B,T,N,1]
        vel = X[..., 2:3]
        dist = X[..., 3:4]
        obs_pair = torch.cat([vel, dist], dim=-1)  # [B,T,N,2]
        c_hat = self.conf_estimator(obs_pair)  # [B,T,N,1]

        outputs_I, alphas = [], []
        for t in range(T):
            h_t = X[:, t, :, :]  # [B,N,F]
            out, att = self.gat(h_t, c_hat[:, t, :, :], dist_mat, A)
            outputs_I.append(out.unsqueeze(1))  # [B,1,N,D]
            alphas.append(att.unsqueeze(1))  # [B,1,N,N]

        H = torch.cat(outputs_I, dim=1)  # [B,T,N,D]

        # Transformer 时间建模
        H_reshape = H.permute(1, 0, 2, 3).reshape(T, B * N, -1)
        H_t = self.temporal_transformer(H_reshape)
        H_t = H_t.reshape(T, B, N, -1).permute(1, 0, 2, 3)

        # 输出
        pred_I = self.fc_I(self.dropout(H_t))  # [B,T,N,1]
        pred_mask = torch.sigmoid(self.fc_mask(self.dropout(H_t)))

        # ★ 暂存注意力与 c_hat（供训练脚本后续正则/可视化使用）
        self.last_alpha = torch.cat(alphas, dim=1)  # [B,T,N,N]
        self.last_c_hat = c_hat  # [B,T,N,1]
        return pred_I, pred_mask

