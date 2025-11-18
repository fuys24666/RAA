import torch
import torch.nn as nn
import torch.nn.functional as F

class RAAGATLayer_NoConf(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.1, alpha=0.2):
        super().__init__()
        self.W = nn.Linear(in_dim, out_dim, bias=False)
        self.a = nn.Linear(out_dim * 2 + 1, 1, bias=False)
        self.leakyrelu = nn.LeakyReLU(alpha)
        self.dropout = nn.Dropout(dropout)


    def forward(self, h, dist_mat, adj):
        """
        h: [B, N, F]
        dist_mat: [N, N]
        adj: [N, N]
        """
        B, N, _ = h.shape
        Wh = self.W(h)  # [B, N, out_dim]

        # --- 全连接展开 ---
        Wh_i = Wh.unsqueeze(2).expand(B, N, N, Wh.size(-1))  # [B,N,N,D]
        Wh_j = Wh.unsqueeze(1).expand(B, N, N, Wh.size(-1))  # [B,N,N,D]

        # 距离 d_ij 展开到 [B,N,N,1]
        d_ij = dist_mat.unsqueeze(0).unsqueeze(-1).expand(B, N, N, 1)

        # 拼接注意力输入（无置信度）
        concat = torch.cat([Wh_i, Wh_j, d_ij], dim=-1)  # [B,N,N,2D+1]
        e = self.leakyrelu(self.a(concat)).squeeze(-1)  # [B,N,N]

        # 用 adj 做 mask（非邻居 = -inf）
        e = e.masked_fill(adj == 0, float('-inf'))

        alpha = torch.softmax(e, dim=-1)  # [B,N,N]
        alpha = self.dropout(alpha)

        # 聚合邻居信息
        h_prime = torch.bmm(alpha, Wh)  # [B, N, out_dim]
        return h_prime, alpha



# ==========================================================
# ✅ RAA-STGNN_NoConf：去置信度但保留 Transformer 时间层
# ==========================================================
class RAASTGNN_NoConf_Transformer(nn.Module):
    def __init__(self, in_dim=7, gcn_dim=64, dropout=0.2, num_heads=2):
        super().__init__()
        self.gat = RAAGATLayer_NoConf(in_dim, gcn_dim, dropout=dropout)

        # 时间 Transformer 与原版保持一致
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=gcn_dim, nhead=num_heads, dim_feedforward=128
        )
        self.temporal_transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        self.fc_I = nn.Linear(gcn_dim, 1)
        self.fc_mask = nn.Linear(gcn_dim, 1)
        self.dropout = nn.Dropout(dropout)
        # ⭐ 新增：用于诊断可视化
        self.last_alpha = None  # [B,T,N,N]

    def forward(self, X, A, dist_mat):
        """
        X: [B, T, N, F]
        A: [N, N]
        dist_mat: [N, N]
        """
        B, T, N, F = X.size()
        outputs_I = []
        alphas = []  # ⭐ 新增

        for t in range(T):
            h_t = X[:, t, :, :]  # 去置信度
            out, att = self.gat(h_t, dist_mat, A)
            outputs_I.append(out.unsqueeze(1))
            alphas.append(att.unsqueeze(1))  # ⭐ 补这一行 [B,1,N,N]

        H = torch.cat(outputs_I, dim=1)  # [B, T, N, gcn_dim]

        # ⭐ 在时间建模前，把所有时间步的 alpha 存起来，供外部画图
        self.last_alpha = torch.cat(alphas, dim=1)  # [B,T,N,N]

        # 时间 Transformer 建模
        H_reshape = H.permute(1, 0, 2, 3).reshape(T, B * N, -1)
        H_t = self.temporal_transformer(H_reshape)
        H_t = H_t.reshape(T, B, N, -1).permute(1, 0, 2, 3)

        # 输出层
        pred_I = self.fc_I(self.dropout(H_t))
        pred_mask = torch.sigmoid(self.fc_mask(self.dropout(H_t)))

        return pred_I, pred_mask





