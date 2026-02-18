import argparse
import torch
import numpy as np
import os
import time
import importlib
import pickle
import matplotlib.pyplot as plt
import torch.nn.functional as F
from tqdm import tqdm
from utils.dataset import TrajectoryDataset
from utils.model import TrajectoryModel

#python test_cum.py --hp_config config/ig_epoch500.py --gpu 0
# -------------------------------
# Evaluation Metrics (train.py 기반)
# -------------------------------
def evaluation_metrics(pred_trajs, gt_trajs):
    gt_ = gt_trajs.unsqueeze(1)
    norm_ = torch.norm(pred_trajs - gt_, dim=-1)
    ade_ = torch.mean(norm_, dim=-1)
    fde_ = norm_[:, :, -1]
    min_ade, _ = torch.min(ade_, dim=-1)
    min_fde, min_fde_index = torch.min(fde_, dim=-1)

    B, K, T, _ = pred_trajs.shape
    min_idx = (
        min_fde_index.unsqueeze(-1)
        .unsqueeze(-1)
        .unsqueeze(-1)
        .expand(-1, 1, T, 2)
    )
    best_preds = torch.gather(pred_trajs, 1, min_idx).squeeze(1)
    jade = torch.norm(best_preds - gt_trajs, dim=-1).mean(dim=1).mean().item()
    jfde = torch.norm(best_preds[:, -1, :] - gt_trajs[:, -1, :], dim=-1).mean().item()

    ade = min_ade.mean().item()
    fde = min_fde.mean().item()
    return ade, fde, jade, jfde, min_fde_index


# -------------------------------
# RMSE 계산
# -------------------------------
def compute_rmse(pred_trajs, gt_trajs, min_fde_index):
    B, K, T, _ = pred_trajs.shape
    best_pred = pred_trajs[torch.arange(B), min_fde_index]  # [B,T,2]
    err = best_pred - gt_trajs
    rmse_x = torch.sqrt(torch.mean(err[..., 0] ** 2, dim=1))
    rmse_y = torch.sqrt(torch.mean(err[..., 1] ** 2, dim=1))
    return rmse_x.mean().item(), rmse_y.mean().item()


# -------------------------------
# 시각화 함수
# -------------------------------
def visualize_prediction_grid(obs_trajs, gt_trajs, pred_trajs, save_path=None, max_samples=3):
    n_samples = min(max_samples, obs_trajs.shape[0])
    plt.figure(figsize=(5 * n_samples, 5))

    for i in range(n_samples):
        plt.subplot(1, n_samples, i + 1)
        obs, gt, preds = obs_trajs[i], gt_trajs[i], pred_trajs[i]
        plt.plot(obs[:, 0], obs[:, 1], 'bo-', label='Observed')
        plt.plot(gt[:, 0], gt[:, 1], 'go-', label='GT')
        for j in range(preds.shape[0]):
            plt.plot(preds[j, :, 0], preds[j, :, 1], '--', alpha=0.6)
        plt.title(f"Sample {i+1}")
        plt.axis("equal")
        plt.legend(fontsize=8)
        plt.grid(True)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


# -------------------------------
# 테스트 함수
# -------------------------------
@torch.no_grad()
def test(model, dataloader, motion_modes, device, visualize=False, save_dir="./visualization"):
    model.eval()
    all_minade = []
    all_fde = []
    ade_sum, fde_sum, jade_sum, jfde_sum = 0, 0, 0, 0
    rmse_x_sum, rmse_y_sum, count = 0, 0, 0
    inference_times = []

    if visualize:
        os.makedirs(save_dir, exist_ok=True)

    for batch_idx, (ped, neis, mask) in enumerate(tqdm(dataloader, desc="Testing")):
        ped, neis, mask = ped.to(device), neis.to(device), mask.to(device)
        ped_obs, gt = ped[:, :args.obs_len], ped[:, args.obs_len:]
        neis_obs = neis[:, :, :args.obs_len]

        torch.cuda.synchronize(); t0 = time.time()
        pred_trajs, scores = model(ped_obs, neis_obs, motion_modes, mask, None, test=True)
        torch.cuda.synchronize(); t1 = time.time()
        inference_times.append(t1 - t0)

        pred_trajs = pred_trajs.reshape(pred_trajs.shape[0], pred_trajs.shape[1], gt.shape[1], 2)

        ade, fde, jade, jfde, min_fde_index = evaluation_metrics(pred_trajs, gt)
        rmse_x, rmse_y = compute_rmse(pred_trajs, gt, min_fde_index)

        # ✅ minADE, FDE 값 저장
        gt_ = gt.unsqueeze(1)
        norm_ = torch.norm(pred_trajs - gt_, dim=-1)
        ade_all = norm_.mean(-1)  # [B,K]
        fde_all = norm_[..., -1]  # [B,K]
        min_ade, _ = torch.min(ade_all, dim=-1)  # [B]
        min_fde, _ = torch.min(fde_all, dim=-1)  # [B]
        all_minade.append(min_ade.cpu())
        all_fde.append(min_fde.cpu())

        ade_sum += ade * ped.shape[0]
        fde_sum += fde * ped.shape[0]
        jade_sum += jade * ped.shape[0]
        jfde_sum += jfde * ped.shape[0]
        rmse_x_sum += rmse_x * ped.shape[0]
        rmse_y_sum += rmse_y * ped.shape[0]
        count += ped.shape[0]

        if visualize and batch_idx < 10:
            save_path = os.path.join(save_dir, f"batch_{batch_idx}.png")
            visualize_prediction_grid(
                ped_obs.cpu().numpy(),
                gt.cpu().numpy(),
                pred_trajs.cpu().numpy(),
                save_path
            )

    # ---------- 평균 계산 ----------
    ade = ade_sum / count
    fde = fde_sum / count
    jade = jade_sum / count
    jfde = jfde_sum / count
    rmse_x = rmse_x_sum / count
    rmse_y = rmse_y_sum / count
    total_time = sum(inference_times)
    infer_time_per_sample = (total_time / count) * 1000

    # ---------- ✅ 추가: Top-K 성능 및 Miss Rate ----------
    all_minade = torch.cat(all_minade)
    all_fde = torch.cat(all_fde)
    n = len(all_minade)
    top1 = int(n * 0.01)
    top2 = int(n * 0.02)
    top3 = int(n * 0.03)

    minade_sorted, _ = torch.sort(all_minade, descending=True)
    top1_ade = minade_sorted[:top1].mean().item()
    top2_ade = minade_sorted[:top2].mean().item()
    top3_ade = minade_sorted[:top3].mean().item()

    miss_rate = (all_fde > 2.0).float().mean().item() * 100

    print(f"ADE: {ade:.4f} | FDE: {fde:.4f} | "
          f"JADE: {jade:.4f} | JFDE: {jfde:.4f} | "
          f"RMSE_X: {rmse_x:.4f} | RMSE_Y: {rmse_y:.4f} | "
          f"Infer: {total_time:.2f}s | {infer_time_per_sample:.2f} ms/sample")
    print(f"Top-1% minADE: {top1_ade:.4f} | Top-2%: {top2_ade:.4f} | Top-3%: {top3_ade:.4f}")
    print(f"Miss Rate (FDE > 2 m): {miss_rate:.2f}%")

    return ade, fde, jade, jfde, rmse_x, rmse_y, total_time, infer_time_per_sample



# -------------------------------
# Main
# -------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, default='./dataset/')
    parser.add_argument('--dataset_name', type=str, default='sliding_window_10')
    parser.add_argument("--hp_config", type=str, default="./config/ig.py")
    parser.add_argument("--checkpoint",type=str,default="/home/user/Algorithm/QCNet_cum/Code/TUTR_Contrastive/checkpoint/vanilla")   
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--obs_len', type=int, default=10)
    parser.add_argument('--pred_len', type=int, default=30)
    parser.add_argument('--num_works', type=int, default=4)
    parser.add_argument('--visualize', action='store_true')
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load config
    spec = importlib.util.spec_from_file_location("hp_config", args.hp_config)
    hp_config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(hp_config)

    # Dataset
    test_dataset = TrajectoryDataset(
        dataset_path=args.dataset_path,
        dataset_name=args.dataset_name,
        dataset_type='test',
        translation=True, rotation=True, scaling=False,
        obs_len=args.obs_len
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, collate_fn=test_dataset.coll_fn,
        batch_size=hp_config.batch_size, shuffle=False,
        num_workers=args.num_works
    )

    # Motion modes
    motion_modes_file = os.path.join(args.dataset_path, f"{args.dataset_name}_motion_modes.pkl")
    with open(motion_modes_file, 'rb') as f:
        motion_modes = pickle.load(f)
    motion_modes = torch.tensor(motion_modes, dtype=torch.float32).to(device)

    # Paths
    ckpt_dir = os.path.join(args.checkpoint, args.dataset_name)
    model_paths = {
        "best": os.path.join(ckpt_dir, "best.pth"),
        # "best_joint": os.path.join(ckpt_dir, "best_joint.pth")
    }

    print("\n================= MODEL COMPARISON =================")
    for label, path in model_paths.items():
        if not os.path.exists(path):
            print(f"⚠️  {path} not found, skipping.")
            continue

        model = TrajectoryModel(
            in_size=2, obs_len=args.obs_len, pred_len=args.pred_len,
            embed_size=hp_config.model_hidden_dim,
            enc_num_layers=2, int_num_layers_list=[1, 1],
            heads=4, forward_expansion=2
        ).to(device)
        model.load_state_dict(torch.load(path, map_location=device))
        model.eval()

        print(f"\n🚀 Evaluating [{args.dataset_name}_{label}] model...")
        test(model, test_loader, motion_modes, device, visualize=args.visualize)
    print("=====================================================")