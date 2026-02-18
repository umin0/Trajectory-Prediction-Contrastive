import argparse
import random
import numpy as np
import torch
import os
import importlib
import time
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from datetime import datetime

from utils.dataset import TrajectoryDataset
from utils.model import TrajectoryModel
from utils.utils import get_motion_modes


# python train.py --dataset_name LIS --hp_config config/ig_epoch100.py --gpu 0 --checkpoint checkpoint/lis/initial_hp_tuning/ --search --search_epochs 80 --search_patience 15

# -------------------------
# Argument parser
# -------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--dataset_path', type=str, default='./dataset/')
parser.add_argument('--dataset_name', type=str)
parser.add_argument("--hp_config", type=str, default=None, help='hyper-parameter')
parser.add_argument('--lr_scaling', action='store_true', default=False)
parser.add_argument('--num_works', type=int, default=2)
parser.add_argument('--obs_len', type=int, default=10)
parser.add_argument('--pred_len', type=int, default=30)
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--data_scaling', type=list, default=[1.9, 0.4])
parser.add_argument('--checkpoint', type=str, default='./checkpoint/vanilla')
args = parser.parse_args()

# -------------------------
# Reproducibility
# -------------------------
seed = args.seed
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

# -------------------------
# Load config
# -------------------------
spec = importlib.util.spec_from_file_location("hp_config", args.hp_config)
hp_config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hp_config)

# -------------------------
# Logging setup
# -------------------------
log_dir = os.path.join(args.checkpoint, args.dataset_name)
os.makedirs(log_dir, exist_ok=True)
log_path = os.path.join(log_dir, "log.txt")
log_file = open(log_path, "a", encoding="utf-8")
log_file.write(f"\n\n===== TRAINING STARTED: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} =====\n")
log_file.flush()

import sys
class Logger(object):
    def __init__(self, stdout, logfile):
        self.terminal = stdout
        self.log = logfile
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
    def flush(self):
        self.terminal.flush()
        self.log.flush()

sys.stdout = Logger(sys.stdout, log_file)
sys.stderr = Logger(sys.stderr, log_file)

print(args)

# -------------------------
# Dataset
# -------------------------
train_dataset = TrajectoryDataset(
    dataset_path=args.dataset_path,
    dataset_name=args.dataset_name,
    dataset_type='train',
    translation=True,
    rotation=True,
    scaling=True,
    obs_len=args.obs_len,
    dist_threshold=hp_config.dist_threshold,
    smooth=False
)

test_dataset = TrajectoryDataset(
    dataset_path=args.dataset_path,
    dataset_name=args.dataset_name,
    dataset_type='val',
    translation=True,
    rotation=True,
    scaling=False,
    obs_len=args.obs_len
)

motion_modes_file = os.path.join(args.dataset_path, f"{args.dataset_name}_motion_modes.pkl")
if not os.path.exists(motion_modes_file):
    print('motion modes generating ...')
    motion_modes = get_motion_modes(train_dataset, args.obs_len, args.pred_len, hp_config.n_clusters,
                                    args.dataset_path, args.dataset_name,
                                    smooth_size=hp_config.smooth_size,
                                    random_rotation=hp_config.random_rotation,
                                    traj_seg=hp_config.traj_seg)
    motion_modes = torch.tensor(motion_modes, dtype=torch.float32).cuda()
else:
    print('motion modes loading ...')
    import pickle
    with open(motion_modes_file, 'rb') as f:
        motion_modes = pickle.load(f)
    motion_modes = torch.tensor(motion_modes, dtype=torch.float32).cuda()

train_loader = DataLoader(train_dataset, collate_fn=train_dataset.coll_fn,
                          batch_size=hp_config.batch_size, shuffle=True, num_workers=args.num_works)
test_loader = DataLoader(test_dataset, collate_fn=test_dataset.coll_fn,
                         batch_size=hp_config.batch_size, shuffle=True, num_workers=args.num_works)

# -------------------------
# Model
# -------------------------
model = TrajectoryModel(in_size=2, obs_len=args.obs_len, pred_len=args.pred_len,
                        embed_size=hp_config.model_hidden_dim,
                        enc_num_layers=2, int_num_layers_list=[1,1],
                        heads=4, forward_expansion=2).cuda()

optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=hp_config.lr)
reg_criterion = torch.nn.SmoothL1Loss().cuda()
cls_criterion = torch.nn.CrossEntropyLoss().cuda()

if args.lr_scaling:
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[270, 400], gamma=0.5)


# -------------------------
# Helper Functions
# -------------------------
def get_cls_label(gt, motion_modes):
    gt = gt.reshape(gt.shape[0], -1).unsqueeze(1)
    motion_modes = motion_modes.reshape(motion_modes.shape[0], -1).unsqueeze(0)
    distance = torch.norm(gt - motion_modes, dim=-1)
    soft_label = F.softmax(-distance, dim=-1)
    closest_mode_indices = torch.argmin(distance, dim=-1)
    return soft_label, closest_mode_indices


def compute_rmse(pred_trajs, gt_trajs):
    """Compute RMSE for X and Y separately"""
    diff = pred_trajs - gt_trajs.unsqueeze(1)  # [B, K, T, 2]
    mse_x = (diff[..., 0] ** 2).mean()
    mse_y = (diff[..., 1] ** 2).mean()
    return torch.sqrt(mse_x).item(), torch.sqrt(mse_y).item()


def train_one_epoch(epoch, model, reg_criterion, cls_criterion, optimizer, dataloader, motion_modes):
    model.train()
    total_loss = []
    for (ped, neis, mask) in dataloader:
        ped, neis, mask = ped.cuda(), neis.cuda(), mask.cuda()
        ped_obs = ped[:, :args.obs_len]
        gt = ped[:, args.obs_len:]
        neis_obs = neis[:, :, :args.obs_len]

        with torch.no_grad():
            soft_label, closest_mode_indices = get_cls_label(gt, motion_modes)

        optimizer.zero_grad()
        pred_traj, scores = model(ped_obs, neis_obs, motion_modes, mask, closest_mode_indices)
        reg_loss = reg_criterion(pred_traj, gt.reshape(pred_traj.shape))
        clf_loss = cls_criterion(scores.squeeze(), soft_label)
        loss = reg_loss + clf_loss
        loss.backward()
        optimizer.step()
        total_loss.append(loss.item())

    return np.mean(total_loss)

# -------------------------
# Test function (ADE/FDE + Joint-ADE/FDE + RMSE_X/Y)
# -------------------------
def test(model, dataloader, motion_modes):
    model.eval()
    ade, fde, jade, jfde, num_traj = 0, 0, 0, 0, 0
    rmse_x_sum, rmse_y_sum = 0, 0
    inference_times = []

    for (ped, neis, mask) in dataloader:
        ped, neis, mask = ped.cuda(), neis.cuda(), mask.cuda()
        ped_obs, gt = ped[:, :args.obs_len], ped[:, args.obs_len:]
        neis_obs = neis[:, :, :args.obs_len]

        torch.cuda.synchronize(); t_start = time.time()
        with torch.no_grad():
            pred_trajs, scores = model(ped_obs, neis_obs, motion_modes, mask, None, test=True)
        torch.cuda.synchronize(); t_end = time.time()
        inference_times.append(t_end - t_start)

        top_k_scores = torch.topk(scores, k=20, dim=-1).values
        # top_k_scores = torch.topk(scores, k=6, dim=-1).values
        top_k_scores = F.softmax(top_k_scores, dim=-1)
        pred_trajs = pred_trajs.reshape(pred_trajs.shape[0], pred_trajs.shape[1], gt.shape[1], 2)

        gt_ = gt.unsqueeze(1)
        norm_ = torch.norm(pred_trajs - gt_, dim=-1)
        ade_ = torch.mean(norm_, dim=-1)
        fde_ = norm_[:, :, -1]
        min_ade, min_ade_index = torch.min(ade_, dim=-1)
        min_fde, min_fde_index = torch.min(fde_, dim=-1)

        batch_index = torch.arange(top_k_scores.shape[0]).cuda()
        min_ade_p = top_k_scores[batch_index, min_ade_index]
        min_fde_p = top_k_scores[batch_index, min_fde_index]
        min_ade = min_ade + (1 - min_ade_p)**2
        min_fde = min_fde + (1 - min_fde_p)**2

        ade += torch.sum(min_ade).item()
        fde += torch.sum(min_fde).item()
        num_traj += ped_obs.shape[0]
        
        B, K, T, _ = pred_trajs.shape
        min_idx = (
            min_fde_index.unsqueeze(-1)
            .unsqueeze(-1)
            .unsqueeze(-1)
            .expand(-1, 1, T, 2)
        )
        # -----------------------------
        # ✅ RMSE 
        # -----------------------------
        best_pred = pred_trajs[torch.arange(B), min_fde_index]  # [B,T,2]
        err = best_pred - gt
        rmse_x = torch.sqrt(torch.mean(err[..., 0]**2, dim=1))
        rmse_y = torch.sqrt(torch.mean(err[..., 1]**2, dim=1))
        rmse_x_sum += rmse_x.sum().item()
        rmse_y_sum += rmse_y.sum().item()

    # -----------------------------
    # ✅ 평균 및 반환
    # -----------------------------
    ade /= num_traj
    fde /= num_traj
    jade /= num_traj
    jfde /= num_traj
    rmse_x = rmse_x_sum / num_traj
    rmse_y = rmse_y_sum / num_traj
    total_infer_time = sum(inference_times)
    infer_time_per_sample = (total_infer_time / num_traj) * 1000

    print(f"ADE: {ade:.4f} | FDE: {fde:.4f} | "
          f"JADE: {jade:.4f} | JFDE: {jfde:.4f} | "
          f"RMSE_X: {rmse_x:.4f} | RMSE_Y: {rmse_y:.4f} | "
          f"Infer: {total_infer_time:.2f}s | {infer_time_per_sample:.2f} ms/sample")

    return ade, fde, rmse_x, rmse_y, num_traj, total_infer_time, infer_time_per_sample

# -------------------------
# Training Loop
# -------------------------
min_ade, min_fde = 99, 99

for ep in range(hp_config.epoch):
    train_loss = train_one_epoch(ep, model, reg_criterion, cls_criterion, optimizer, train_loader, motion_modes)
    ade, fde, rmse_x, rmse_y, num_traj, total_infer_time, infer_time_per_sample = test(model, test_loader, motion_modes)

    if args.lr_scaling: scheduler.step()

    # best ADE+FDE
    if min_fde + min_ade > ade + fde:
        min_fde, min_ade = fde, ade
        torch.save(model.state_dict(), os.path.join(log_dir, 'best.pth'))
        print(f"[Epoch {ep}] ✅ best.pth updated (ADE+FDE improved)")

    print(f"[Epoch {ep:03d}] Loss: {train_loss:.4f} | "
      f"ADE: {ade:.4f} | FDE: {fde:.4f} | "
      f"RMSE_X: {rmse_x:.4f} | RMSE_Y: {rmse_y:.4f} | "
      f"Infer: {total_infer_time:.2f}s | {infer_time_per_sample:.2f} ms/sample")


# -------------------------
# Finish
# -------------------------
log_file.write(f"\n===== TRAINING FINISHED: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} =====\n")
log_file.flush()
log_file.close()
