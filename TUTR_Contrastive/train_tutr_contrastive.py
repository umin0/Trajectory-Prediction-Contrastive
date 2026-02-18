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
from utils.model_contrative import TrajectoryModel
from utils.utils_contrastive import get_motion_modes, difficulty_contrastive_loss  
from itertools import product
from filterpy.kalman import KalmanFilter


"""
python train.py \
  --dataset_name ETH \
  --hp_config ./ig.py \
  --gpu 0

python train.py --dataset_path dataset/initial/ --dataset_name lis --hp_config config/ig.py --gpu 0 --checkpoint checkpoint/lis/initial_hp_tuning/ --search --search_epochs 80 --search_patience 15
"""
# python train_contrastive_v2.py --dataset_name LIS --hp_config config/ig_contrastive.py --gpu 0 --checkpoint checkpoint/lis/contrastive_hp_tuning/ --search --search_epochs 80 --search_patience 15


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
parser.add_argument('--checkpoint', type=str, default='./checkpoint/contrastive')


parser.add_argument('--search', action='store_true', default=False,
                    help='run hyper-parameter search instead of single training')
parser.add_argument('--search_epochs', type=int, default=80,
                    help='number of epochs for each config in search mode')
parser.add_argument('--search_patience', type=int, default=15,
                    help='early stopping patience in search mode')

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
# Load config (ig.py)
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
print(f"Loaded hp_config from: {args.hp_config}")
print(f"Initial hp_config: lr={hp_config.lr}, batch_size={hp_config.batch_size}, "
      f"n_clusters={hp_config.n_clusters}")

# ============================================================
# 공통 함수들 (데이터셋, 모션 모드, 모델 생성)
# ============================================================

def build_motion_modes(hp_config, args):
    """
    n_clusters가 바뀔 수 있으므로, 파일 이름에 n_clusters를 포함.
    예: datasetname_motion_modes_80.pkl
    """
    motion_modes_file = os.path.join(
        args.dataset_path,
        f"{args.dataset_name}_motion_modes_{hp_config.n_clusters}.pkl"
    )

    # 여기서 train_dataset 만들고, positional 인자로 넘김
    train_dataset = train_dataset_for_modes(args, hp_config)

    if not os.path.exists(motion_modes_file):
        print(f"[n_clusters={hp_config.n_clusters}] motion modes generating ...")
        motion_modes = get_motion_modes(
            train_dataset,              # 첫 번째 인자는 dataset
            args.obs_len,               # obs_len (positional)
            args.pred_len,              # pred_len (positional)
            hp_config.n_clusters,       # n_clusters (positional)
            args.dataset_path,
            args.dataset_name,
            smooth_size=hp_config.smooth_size,
            random_rotation=hp_config.random_rotation,
            traj_seg=hp_config.traj_seg
        )
        import pickle
        with open(motion_modes_file, 'wb') as f:
            pickle.dump(motion_modes, f)
    else:
        print(f"[n_clusters={hp_config.n_clusters}] motion modes loading ...")
        import pickle
        with open(motion_modes_file, 'rb') as f:
            motion_modes = pickle.load(f)

    motion_modes = torch.tensor(motion_modes, dtype=torch.float32).cuda()
    return motion_modes


def train_dataset_for_modes(args, hp_config):
    """
    motion_modes 생성에 사용할 train_dataset.
    (dist_threshold, smooth_size 등 하이퍼파라미터)
    """
    dataset = TrajectoryDataset(
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
    return dataset


def build_dataloaders(hp_config, args):
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

    train_loader = DataLoader(
        train_dataset,
        collate_fn=train_dataset.coll_fn,
        batch_size=hp_config.batch_size,
        shuffle=True,
        num_workers=args.num_works
    )
    test_loader = DataLoader(
        test_dataset,
        collate_fn=test_dataset.coll_fn,
        batch_size=hp_config.batch_size,
        shuffle=True,
        num_workers=args.num_works
    )
    return train_loader, test_loader


def build_model_and_optim(hp_config, args):
    model = TrajectoryModel(
        in_size=2,
        obs_len=args.obs_len,
        pred_len=args.pred_len,
        embed_size=hp_config.model_hidden_dim,
        enc_num_layers=2,
        int_num_layers_list=[1, 1],
        heads=4,
        forward_expansion=2
    ).cuda()

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=hp_config.lr
    )

    if args.lr_scaling:
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[270, 400],
            gamma=0.5
        )
    else:
        scheduler = None

    reg_criterion = torch.nn.SmoothL1Loss().cuda()
    cls_criterion = torch.nn.CrossEntropyLoss().cuda()

    return model, optimizer, scheduler, reg_criterion, cls_criterion

# ============================================================
# 기존 helper 함수 (공통 함수)
# ============================================================

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


def compute_kf_difficulty(ped_obs, gt, dt=0.1):
    """
    ped_obs: [B, 10, 2]
    gt:      [B, 30, 2]

    """

    with torch.no_grad():
        device = ped_obs.device
        B = ped_obs.shape[0]

        difficulties = []

        obs_np = ped_obs.detach().cpu().numpy()
        gt_np  = gt.detach().cpu().numpy()

        for i in range(B):
            xy = np.concatenate([obs_np[i], gt_np[i]], axis=0)  # 40프레임

            if xy.shape[0] < 10:  # QCNet min_len 유지
                difficulties.append(0.0)
                continue

            kf = KalmanFilter(dim_x=4, dim_z=2)

            kf.F = np.array([[1, 0, dt, 0],
                             [0, 1, 0, dt],
                             [0, 0, 1,  0],
                             [0, 0, 0,  1]])

            kf.H = np.array([[1, 0, 0, 0],
                             [0, 1, 0, 0]])

            kf.P *= 10
            kf.R *= 0.1
            kf.Q = np.eye(4) * 0.01
            kf.x = np.array([xy[0, 0], xy[0, 1], 0, 0])

            preds = []
            for t in range(1, len(xy)):
                kf.predict()
                preds.append(kf.x[:2].copy())  # update 전 저장
                kf.update(xy[t])

            preds = np.array(preds)

            err = np.linalg.norm(preds - xy[1:], axis=1).mean()
            difficulties.append(float(err))

        return torch.tensor(difficulties, dtype=torch.float32, device=device)



def train_one_epoch(epoch, model, reg_criterion, cls_criterion,
                    optimizer, dataloader, motion_modes, hp_config):
    model.train()
    total_loss = []

    for (ped, neis, mask) in dataloader:
        ped, neis, mask = ped.cuda(), neis.cuda(), mask.cuda()
        ped_obs = ped[:, :args.obs_len]
        gt = ped[:, args.obs_len:]
        neis_obs = neis[:, :, :args.obs_len]

        # ----- difficulty, motion-mode soft label -----
        with torch.no_grad():
            soft_label, closest_mode_indices = get_cls_label(gt, motion_modes)
            difficulty = compute_kf_difficulty(ped_obs, gt)  # [B]

        optimizer.zero_grad()

        pred_traj, scores, bottleneck = model(
            ped_obs, neis_obs, motion_modes, mask, closest_mode_indices,
            return_feat=True   # <-- model.py에서 이 플래그 처리
        )

        # ===============================
        # 2) 기본 loss (regression + classification)
        # ===============================
        reg_loss = reg_criterion(pred_traj, gt.reshape(pred_traj.shape))
        clf_loss = cls_criterion(scores.squeeze(), soft_label)

        # ===============================
        # 3) difficulty-aware contrastive
        # ===============================
        if hp_config.lambda_contr > 0.0:
            z = bottleneck  # [B, D]
            contr_loss = difficulty_contrastive_loss(
                z,
                difficulty,
                tau=hp_config.contr_tau,
                pos_delta=hp_config.contr_pos_delta,
                neg_delta=hp_config.contr_neg_delta,
            )
            loss = reg_loss + clf_loss + hp_config.lambda_contr * contr_loss

            if torch.rand(1).item() < 0.01:  
                print(f"reg: {reg_loss.item():.4f}, "
                    f"cls: {clf_loss.item():.4f}, "
                    f"contr: {contr_loss.item():.4f}, "
                    f"total: {loss.item():.4f}")
        else:
            loss = reg_loss + clf_loss

        loss.backward()
        optimizer.step()
        total_loss.append(loss.item())

    return np.mean(total_loss)


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
        
        # RMSE
        best_pred = pred_trajs[torch.arange(B), min_fde_index]  # [B,T,2]
        err = best_pred - gt
        rmse_x = torch.sqrt(torch.mean(err[..., 0]**2, dim=1))
        rmse_y = torch.sqrt(torch.mean(err[..., 1]**2, dim=1))
        rmse_x_sum += rmse_x.sum().item()
        rmse_y_sum += rmse_y.sum().item()

    ade /= num_traj
    fde /= num_traj
    rmse_x = rmse_x_sum / num_traj
    rmse_y = rmse_y_sum / num_traj
    total_infer_time = sum(inference_times)
    infer_time_per_sample = (total_infer_time / num_traj) * 1000

    print(f"ADE: {ade:.4f} | FDE: {fde:.4f} | "
          f"RMSE_X: {rmse_x:.4f} | RMSE_Y: {rmse_y:.4f} | "
          f"Infer: {total_infer_time:.2f}s | {infer_time_per_sample:.2f} ms/sample")

    return ade, fde, rmse_x, rmse_y, num_traj, total_infer_time, infer_time_per_sample

# ============================================================
# 공통 training 루프 (config 하나에 대해)
# ============================================================

def run_training(hp_config, args, max_epochs=None, save_best=True, tag=""):
    """
    hp_config: ig.py 모듈 (속성 값을 바꾸면서 사용)
    max_epochs: None면 hp_config.epoch 사용, 아니면 지정한 값 사용
    save_best: True면 best.pth / best_joint.pth 저장
    tag: search용 config 구분을 위한 (파일명 suffix)
    """
    # 데이터, 모션 모드 준비
    train_loader, test_loader = build_dataloaders(hp_config, args)
    motion_modes = build_motion_modes(hp_config, args)

    # 모델 / 옵티마이저 / 스케줄러 / 손실함수 준비
    model, optimizer, scheduler, reg_criterion, cls_criterion = build_model_and_optim(hp_config, args)

    min_ade, min_fde = 99, 99
    best_metric = min_ade + min_fde

    if max_epochs is None:
        total_epochs = hp_config.epoch
    else:
        total_epochs = max_epochs

    print(
        f"==> Training with config:\n"
        f"  lr={hp_config.lr}\n"
        f"  batch_size={hp_config.batch_size}\n"
        f"  n_clusters={hp_config.n_clusters}\n"
        f"  epochs={total_epochs}\n"
        f"  lambda_contr={hp_config.lambda_contr}\n"
        f"  contr_tau={hp_config.contr_tau}\n"
        f"  contr_pos_delta={hp_config.contr_pos_delta}\n"
        f"  contr_neg_delta={hp_config.contr_neg_delta}"
    )


    for ep in range(total_epochs):
        train_loss = train_one_epoch(ep, model, reg_criterion, cls_criterion, optimizer, train_loader, motion_modes, hp_config)
        ade, fde, rmse_x, rmse_y, num_traj, total_infer_time, infer_time_per_sample = test(model, test_loader, motion_modes)

        if args.lr_scaling and scheduler is not None:
            scheduler.step()

        # best ADE+FDE
        if min_fde + min_ade > ade + fde:
            min_fde, min_ade = fde, ade
            best_metric = min_ade + min_fde
            if save_best:
                suffix = f"_{tag}" if tag else ""
                torch.save(model.state_dict(), os.path.join(log_dir, f'best{suffix}.pth'))
                print(f"[Epoch {ep}] -> best{suffix}.pth updated (ADE+FDE improved)")


        print(f"[Epoch {ep:03d}] Loss: {train_loss:.4f} | "
              f"ADE: {ade:.4f} | FDE: {fde:.4f} | "
              f"RMSE_X: {rmse_x:.4f} | RMSE_Y: {rmse_y:.4f} | "
              f"Infer: {total_infer_time:.2f}s | {infer_time_per_sample:.2f} ms/sample")

    return best_metric, (min_ade, min_fde)

# ============================================================
# 하이퍼파라미터 서치
# ============================================================

def hyperparam_search(hp_config, args):
    """
    여러 하이퍼파라미터 조합에 대해 search_epochs만큼 학습 후 ADE+FDE 기준으로 best config 탐색
    """
    search_space = hp_config.search_space  # ig.py에서 가져오기
    keys = list(search_space.keys())
    values_list = [search_space[k] for k in keys]

    best_overall_metric = float('inf')
    best_cfg = None  # (key -> value) 딕셔너리로 저장

    for combo in product(*values_list):
        # hp_config에 값 주입
        cfg_dict = {}
        print("\n" + "=" * 80)
        print("[SEARCH] Trying config:")
        for k, v in zip(keys, combo):
            setattr(hp_config, k, v)   # hp_config.lr, hp_config.batch_size, ...
            cfg_dict[k] = v
            print(f"  {k}: {v}")
        print("=" * 80)

        # config 이름 태그용 문자열
        tag_str = "_".join(f"{k}{v}" for k, v in cfg_dict.items())

        # search_epochs 동안만 학습
        metric, _ = run_training(
            hp_config, args,
            max_epochs=args.search_epochs,
            save_best=False,
            tag=f"search_{tag_str}"
        )

        print(f"[SEARCH RESULT] {cfg_dict} -> (ADE+FDE)={metric:.4f}")

        if metric < best_overall_metric:
            best_overall_metric = metric
            best_cfg = cfg_dict

    print("\n" + "=" * 80)
    print(f"[SEARCH DONE] Best config: {best_cfg} | metric={best_overall_metric:.4f}")
    print("=" * 80)

    # hp_config를 best config로 세팅
    for k, v in best_cfg.items():
        setattr(hp_config, k, v)

    return best_cfg, best_overall_metric

# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    if args.search:
        # 1) 하이퍼파라미터 서치
        best_cfg, best_metric = hyperparam_search(hp_config, args)

        # 2) best config로 full training (hp_config.epoch 사용)
        print("\n[FULL TRAIN] Start with best config from search ...")
        best_metric_full, best_details = run_training(
            hp_config, args,
            max_epochs=None,   # hp_config.epoch 사용
            save_best=True,
            tag="final"
        )
        print(f"[FULL TRAIN DONE] metric={best_metric_full:.4f}, details={best_details}")
    else:
        # 0) 하이퍼파라미터 서치 없이 full training
        best_metric_full, best_details = run_training(
            hp_config, args,
            max_epochs=None,
            save_best=True,
            tag=""
        )
        print(f"[TRAIN DONE] metric={best_metric_full:.4f}, details={best_details}")

    # -------------------------
    # Finish
    # -------------------------
    log_file.write(f"\n===== TRAINING FINISHED: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} =====\n")
    log_file.flush()
    log_file.close()
