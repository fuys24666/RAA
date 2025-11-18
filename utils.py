import torch

def early_hit_accuracy(pred_mask, true_mask, early_window=10):
    """
    计算预测波到达的准确率。
    若模型在波实际到达前 early_window 帧内预测到，则认为预测成功。
    支持输入形状：(B, T, N)
    """
    with torch.no_grad():
        if pred_mask.ndim == 2:  # 兼容旧版本
            pred_mask = pred_mask.unsqueeze(1)
            true_mask = true_mask.unsqueeze(1)

        B, T, N = pred_mask.shape
        correct, total = 0, 0

        for b in range(B):
            for n in range(N):
                true_idx = (true_mask[b, :, n] > 0.5).nonzero(as_tuple=True)[0]
                pred_idx = (pred_mask[b, :, n] > 0.5).nonzero(as_tuple=True)[0]
                if len(true_idx) > 0 and len(pred_idx) > 0:
                    t_hit = true_idx[0].item()
                    t_pred = pred_idx[0].item()
                    if (t_pred >= t_hit - early_window) and (t_pred <= t_hit):
                        correct += 1
                    total += 1
        return correct / total if total > 0 else 0.0
