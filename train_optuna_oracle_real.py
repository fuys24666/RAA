import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import optuna
import pandas as pd


from model_raa_oracle import RAASTGNN_oracle as STGNNModel
from utils import early_hit_accuracy

def set_seed(seed):
    import random
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # 保证确定性（CPU/GPU）
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# -----------------------------
# 通用日志写入函数
# -----------------------------
def log_to_csv(entry: dict, path="logs/training_log_raa_oracle_real_10_2000_seed42.csv"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df = pd.DataFrame([entry])
    if os.path.exists(path):
        df.to_csv(path, mode="a", header=False, index=False)
    else:
        df.to_csv(path, index=False)


# -----------------------------
# 数据加载
# -----------------------------
def load_dataset(split="train"):
    folder = "data/generated_2000"
    files = [f for f in os.listdir(folder) if split in f]
    X_list, y_mask_list, impact_list = [], [], []

    for f in files:
        data = np.load(os.path.join(folder, f))
        X = torch.tensor(data["X"], dtype=torch.float32)         # [1,T,N,6]
        y = torch.tensor(data["wave_mask"], dtype=torch.float32) # [1,T,N]

        # 使用现实版 heuristic 置信度
        if "conf_real" in data.files:
            c_real = torch.tensor(data["conf_real"], dtype=torch.float32).unsqueeze(-1)  # [1,T,N,1]
            # 布局保持一致：[posx,posy,vel,dist,intensity,conf_real,message]
            X = torch.cat([X[..., :5], c_real, X[..., 5:6]], dim=-1)  # → [1,T,N,7]

        impact = torch.tensor(data["impact_intensity"], dtype=torch.float32)  # [1,N,1]

        X_list.append(X)
        y_mask_list.append(y)
        impact_list.append(impact)

    X = torch.cat(X_list, dim=0)
    y_mask = torch.cat(y_mask_list, dim=0)
    y_impact = torch.cat(impact_list, dim=0)
    return TensorDataset(X, y_mask, y_impact)


def dump_diagnostics(model, loader, A, device, dist_mat, out_path):
    """
    跑一遍 loader，把方便画图的数据都存到 npz 里
    """
    model.eval()
    all_X = []
    all_y_mask = []
    all_y_impact = []

    all_I_pred = []
    all_mask_pred = []
    all_alpha = []

    # 只有 self / oracle 有 conf 相关（oracle: 真实 conf 在 X 里，self: c_hat）
    all_c_hat = []   # 只在 self 版本里真正用
    # 如果你后面想加 true_conf，可以再扩展

    with torch.no_grad():
        dist_norm = dist_mat / (dist_mat.max() + 1e-6)

        for X, y_mask, y_impact in loader:
            X = X.to(device)
            y_mask = y_mask.to(device)
            y_impact = y_impact.to(device)

            pred_I, pred_mask = model(X, A, dist_norm)  # [B,T,N,1], [B,T,N,1]
            # 统一到 cpu numpy，方便之后用 numpy/matplotlib 画图
            all_X.append(X.cpu().numpy())
            all_y_mask.append(y_mask.cpu().numpy())
            all_y_impact.append(y_impact.cpu().numpy())

            all_I_pred.append(pred_I.cpu().numpy())          # [B,T,N,1]
            all_mask_pred.append(pred_mask.cpu().numpy())    # [B,T,N,1]

            # 注意：这里用了我们在模型里存的 last_alpha / last_c_hat
            if getattr(model, "last_alpha", None) is not None:
                all_alpha.append(model.last_alpha.cpu().numpy())  # [B,T,N,N]

            if getattr(model, "last_c_hat", None) is not None:
                all_c_hat.append(model.last_c_hat.cpu().numpy())  # [B,T,N,1]

    # 拼起来
    X_all = np.concatenate(all_X, axis=0)              # [S,T,N,F]
    y_mask_all = np.concatenate(all_y_mask, axis=0)    # [S,T,N]
    y_impact_all = np.concatenate(all_y_impact, axis=0)  # [S,N,1]
    I_pred_all = np.concatenate(all_I_pred, axis=0)    # [S,T,N,1]
    mask_pred_all = np.concatenate(all_mask_pred, axis=0)  # [S,T,N,1]

    alpha_all = np.concatenate(all_alpha, axis=0) if all_alpha else None
    c_hat_all = np.concatenate(all_c_hat, axis=0) if all_c_hat else None

    np.savez(out_path,
             X=X_all,
             y_mask=y_mask_all,
             y_impact=y_impact_all,
             I_pred=I_pred_all,
             mask_pred=mask_pred_all,
             alpha=alpha_all,
             c_hat=c_hat_all)
    print(f"[DIAG] Saved diagnostics to {out_path}")



# -----------------------------
# 模型评估
# -----------------------------
def evaluate(model, loader, A, device, dist_mat):
    model.eval()
    mse_fn = nn.MSELoss()
    mse_total, acc_total, count = 0, 0, 0
    with torch.no_grad():
        dist_norm = dist_mat / (dist_mat.max() + 1e-6)
        for X, y_mask, y_impact in loader:
            X = X.to(device)
            y_mask = y_mask.to(device)
            y_impact = y_impact.to(device)

            pred_I, pred_mask = model(X, A, dist_norm)  # pred_I: [B,T,N,1]
            pred_I = pred_I.squeeze(-1)  # [B,T,N]
            pred_last = pred_I[:, -1, :]  # [B,N]
            target_impact = y_impact.squeeze(-1)        # [B,N]

            mse = mse_fn(pred_last, target_impact)
            acc = early_hit_accuracy(pred_mask.squeeze(-1), y_mask)

            mse_total += mse.item() * X.size(0)
            acc_total += acc * X.size(0)
            count += X.size(0)
    return mse_total / count, acc_total / count


# -----------------------------
# 训练 + 验证 + 记录
# -----------------------------
def train_eval(cfg, train_data, val_data, A, device, dist_mat, trial_id=None):
    train_loader = DataLoader(train_data, batch_size=cfg["batch_size"], shuffle=True)
    val_loader = DataLoader(val_data, batch_size=cfg["batch_size"], shuffle=False)

    sample_X, _, _ = next(iter(train_loader))
    input_dim = sample_X.shape[-1]

    model = STGNNModel(
        in_dim=input_dim,
        gcn_dim=cfg["gat_dim"],
        gru_dim=cfg["tf_dim"],
        dropout=cfg["dropout"],
        num_heads=cfg["num_heads"]
    ).to(device)

    print(f"[INFO] Using RAA-oracle (gat_dim={cfg['gat_dim']}, tf_dim={cfg['tf_dim']})")

    optimizer = optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    # ✅ Warmup + CosineDecay 调度器
    from torch.optim.lr_scheduler import LambdaLR

    def lr_lambda(epoch):
        warmup_epochs = 5  # RAA推荐5轮升温
        if epoch < warmup_epochs:
            return float(epoch + 1) / float(warmup_epochs)
        else:
            progress = (epoch - warmup_epochs) / float(cfg["epochs"] - warmup_epochs)
            return 0.5 * (1 + np.cos(np.pi * progress)) * 0.9 + 0.1

    scheduler = LambdaLR(optimizer, lr_lambda)

    mse_fn = nn.MSELoss()
    bce_fn = nn.BCELoss()

    start_time = time.time()
    for epoch in range(cfg["epochs"]):
        epoch_start = time.time()
        model.train()
        total_loss = 0
        for X, y_mask, y_impact in train_loader:
            X = X.to(device)
            y_mask = y_mask.to(device)
            y_impact = y_impact.to(device)

            dist_norm = dist_mat / (dist_mat.max() + 1e-6)
            pred_I, pred_mask = model(X, A, dist_norm)

            pred_I = pred_I.squeeze(-1)  # [B,T,N]
            pred_last = pred_I[:, -1, :]  # [B,N]

            target_impact = y_impact.squeeze(-1)  # [B,N]

            loss_mse = mse_fn(pred_last, target_impact)
            loss_bce = bce_fn(pred_mask.squeeze(-1), y_mask)
            loss = loss_mse + 0.5 * loss_bce

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        epoch_time = time.time() - epoch_start
        val_mse, val_acc = evaluate(model, val_loader, A, device, dist_mat)
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        print(f"  [Epoch {epoch:02d}] TrainLoss={total_loss/len(train_loader):.4f} "
              f"ValMSE={val_mse:.4f} ValAcc={val_acc:.4f} LR={current_lr:.6f} EpochTime={epoch_time:.2f}s")

        # 🔹 写入 epoch 记录
        log_to_csv({
            "trial": trial_id,
            "phase": "epoch",
            "epoch": epoch,
            "train_loss": total_loss / len(train_loader),
            "val_mse": val_mse,
            "val_acc": val_acc,
            "epoch_time_s": epoch_time,
            "train_time_s": None,
            "total_time_s": None,
            "gat_dim": cfg["gat_dim"],
            "tf_dim": cfg["tf_dim"],
            "num_heads": cfg["num_heads"],
            "dropout": cfg["dropout"],
            "lr": scheduler.get_last_lr()[0],  # ✅ 实时学习率
            "batch_size": cfg["batch_size"],   # ✅ 记录实际batch
            "weight_decay": cfg["weight_decay"]
        })

    total_time = time.time() - start_time
    print(f"[Trial {trial_id}] Total training time: {total_time:.2f}s")
    return val_mse, val_acc, model, total_time


# -----------------------------
# Optuna 调参目标函数
# -----------------------------
def objective(trial):
    # 让 trial 具有可重复性，但不同 trial 使用不同 seed
    set_seed(1000 + trial.number)
    print(f"\n→ Starting Trial {trial.number}")
    trial_start = time.time()

    cfg = {
        "gat_dim": trial.suggest_categorical("gat_dim", [32, 64, 128]),
        "tf_dim": trial.suggest_categorical("tf_dim", [64, 128, 256]),
        "num_heads": trial.suggest_categorical("num_heads", [2, 4]),
        "dropout": trial.suggest_float("dropout", 0.1, 0.5),
        "lr": trial.suggest_float("lr", 3e-4, 2e-3, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64]),
        "epochs": 20,
    }

    device = torch.device("cpu")
    train_data = load_dataset("train")
    val_data = load_dataset("val")

    from torch.utils.data import DataLoader
    import os, glob
    import numpy as np
    from scipy.spatial.distance import cdist

    # 1) 任选一个训练样本文件，读取 positions
    sample_npz = sorted(glob.glob(os.path.join("data", "generated_2000", "*train*.npz")))[0]
    pos = np.load(sample_npz)["positions"]  # shape: [N, 2]
    N = pos.shape[0]

    # 2) 构建邻接 A（全连接，主对角=0）
    A = torch.ones(N, N, device=device)
    A.fill_diagonal_(0)

    # 3) 用 positions 计算真实距离并归一化到 [0,1]
    dist_np = cdist(pos, pos).astype(np.float32)
    dist_mat = torch.tensor(dist_np, dtype=torch.float32, device=device)
    dist_mat = dist_mat / (dist_mat.max() + 1e-6)

    # （可选）调试打印
    print("dist_mat stats:", dist_mat.min().item(), dist_mat.max().item(), dist_mat.mean().item())

    val_mse, val_acc, _, train_time = train_eval(cfg, train_data, val_data, A, device, dist_mat, trial.number)
    total_time = time.time() - trial_start

    print(f"[Trial {trial.number:02d}] val_acc={val_acc:.4f}, val_mse={val_mse:.4f}, "
          f"train_time={train_time:.2f}s, total_time={total_time:.2f}s")

    # 🔹 写入最终结果（同一 CSV）
    log_to_csv({
        "trial": trial.number,
        "phase": "final",
        "epoch": None,
        "train_loss": None,
        "val_mse": val_mse,
        "val_acc": val_acc,
        "epoch_time_s": None,
        "train_time_s": train_time,
        "total_time_s": total_time,
        "gat_dim": cfg.get("gat_dim"),
        "tf_dim": cfg.get("tf_dim"),
        "num_heads": cfg.get("num_heads"),
        "dropout": cfg.get("dropout"),
        "lr": cfg.get("lr"),
        "batch_size": cfg.get("batch_size"),
        "weight_decay": cfg.get("weight_decay"),
    })

    return 1 - val_acc


# -----------------------------
# 主流程
# -----------------------------
def main():
    # 主实验的固定种子（用于论文 Table 1 的结果）
    set_seed(42)

    print("[OK] Loaded datasets and model")
    print("=== Running Optuna search (CPU) ===")

    study_start = time.time()
    study = optuna.create_study(direction="minimize")

    def logging_callback(study, trial):
        print(f"[Trial {trial.number:02d}] val_acc={1 - trial.value:.4f} | params={trial.params}")
        print("-" * 80)

    study.optimize(objective, n_trials=20, callbacks=[logging_callback], catch=(Exception,))

    best_cfg = study.best_params
    print(f"Best config: {best_cfg}")

    device = torch.device("cpu")

    # 1) 读数据集
    train_data = load_dataset("train")
    val_data = load_dataset("val")
    test_data = load_dataset("test")

    # 2) 创建 DataLoader（先有 loader，才能 sample 一个 batch）
    bs = int(best_cfg.get("batch_size", 16))
    train_loader = DataLoader(train_data, batch_size=bs, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=bs, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=bs, shuffle=False)

    # 3) 从一个 batch 里取样，得到 N（[B, T, N, F]）
    sample_X, _, _ = next(iter(train_loader))
    N = sample_X.shape[2]

    # 4) 动态构建图（全连接，主对角为 0），并放到 device
    A = torch.ones(N, N, device=device)
    A.fill_diagonal_(0)

    # 5) 距离矩阵：从 npz 文件读取 positions 计算真实几何距离
    from scipy.spatial.distance import cdist
    import numpy as np

    sample_file = "data/generated_2000/snapshot_task1_run000_train.npz"  # 可以换成任意一个 train 文件
    data_npz = np.load(sample_file)
    pos = data_npz["positions"]  # shape [N, 2]
    dist_np = cdist(pos, pos)  # 欧氏距离矩阵
    dist_mat = torch.tensor(dist_np, dtype=torch.float32, device=device)
    dist_mat = dist_mat / (dist_mat.max() + 1e-6)

    # ✅ 在这里打印统计信息
    print("dist_mat stats:",
          dist_mat.min().item(),
          dist_mat.max().item(),
          dist_mat.mean().item())

    print("\n=== Retraining with best config ===")
    best_cfg["epochs"] = 60

    _, _, best_model, train_time = train_eval(best_cfg, train_data, val_data, A, device, dist_mat, trial_id="best")

    print("\n=== Final evaluation on test set ===")
    test_loader = DataLoader(test_data, batch_size=best_cfg["batch_size"], shuffle=False)
    test_mse, test_acc = evaluate(best_model, test_loader, A, device, dist_mat)
    print(f"Test MSE={test_mse:.4f}, Test Acc={test_acc:.4f}")

    torch.save({
        "model_state_dict": best_model.state_dict(),
        "config": best_cfg,
        "test_mse": test_mse,
        "test_acc": test_acc
    }, "logs/best_stgnn_raa_oracle_real_10_2000_seed42.pt")

    print(f"[OK] Saved full model checkpoint → best_stgnn_raa_oracle_real.pt")
    print(f"Total study time: {(time.time() - study_start)/60:.2f} min")
    # ⭐ 额外：对 test_loader 做一次全量诊断 dump（方便后续画图）
    diag_path = "logs/diagnostics_oracle_real_10_2000_seed42.npz"
    dump_diagnostics(best_model, test_loader, A, device, dist_mat, diag_path)


if __name__ == "__main__":
    main()