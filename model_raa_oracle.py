import torch
import torch.nn as nn
import torch.nn.functional as F


# ==========================================================
# Reliability-Aware Graph Attention Layer (原版保持)
# ==========================================================
class RAAGATLayer_oracle(nn.Module):
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


# ==========================================================
# ✅ RAA-GAT + Transformer with Confidence Rescaling
# ==========================================================
class RAASTGNN_oracle(nn.Module):
    def __init__(self, in_dim=7, gcn_dim=64, gru_dim=64, dropout=0.2, num_heads=2):
        super().__init__()
        self.gat = RAAGATLayer_oracle(in_dim, gcn_dim, dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=gcn_dim, nhead=num_heads, dim_feedforward=128
        )
        self.temporal_transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.fc_I = nn.Linear(gcn_dim, 1)
        self.fc_mask = nn.Linear(gcn_dim, 1)
        self.dropout = nn.Dropout(dropout)

        self.last_alpha = None

    def forward(self, X, A, dist_mat):
        """
        X: [B, T, N, F]
        A: [N, N]
        dist_mat: [N, N]
        """
        B, T, N, F = X.size()

        # 现在（建议保留 [0,1]）：
        conf = X[..., 5:6]  # [B,T,N,1]

        outputs_I = []
        alphas = []  # ⭐ 新增

        for t in range(T):
            h_t = X[:, t, :, :]
            out, att = self.gat(h_t, conf[:, t, :, :], dist_mat, A)
            outputs_I.append(out.unsqueeze(1))
            alphas.append(att.unsqueeze(1))  # ⭐ 补这一行 [B,1,N,N]

        H = torch.cat(outputs_I, dim=1)  # [B, T, N, gcn_dim]
        self.last_alpha = torch.cat(alphas, dim=1)  # [B,T,N,N]
        # 时间注意力（Transformer）
        H_reshape = H.permute(1, 0, 2, 3).reshape(T, B * N, -1)
        H_t = self.temporal_transformer(H_reshape)
        H_t = H_t.reshape(T, B, N, -1).permute(1, 0, 2, 3)

        # 输出层
        pred_I = self.fc_I(self.dropout(H_t))
        pred_mask = torch.sigmoid(self.fc_mask(self.dropout(H_t)))

        return pred_I, pred_mask
