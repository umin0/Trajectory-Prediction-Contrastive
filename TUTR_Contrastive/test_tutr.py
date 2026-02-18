# -*- coding: utf-8 -*-
"""
test_csv_meta_scenario.py (중복 예측 제거 + 디렉토리 모드 지원 버전, FDE-min 모드 선택)

- 원본 scenario CSV (world 좌표)를 읽어서
- 모델이 학습할 때와 동일하게 local 좌표로 변환해서 입력 만들고
- 모델 예측(local)을 다시 world로 되돌린 뒤
- **입력 scenario CSV와 동일한 포맷 + pred_x, pred_y 컬럼만 추가**해서 저장
- 각 track_id에 대해 딱 한 번만 (obs_len + pred_len) 윈도우를 사용하므로
  (track_id, frame_id) 에 대해 중복 예측이 생기지 않음
- csv_path 가 단일 파일이면 해당 파일만,
  디렉토리이면 하위의 모든 *.csv 파일에 대해 위 과정을 수행
"""
from tqdm import tqdm
import argparse
import os
import importlib.util
import pickle
from datetime import datetime
import glob  # 디렉토리 모드용

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

SCALE_X, SCALE_Y = 1.9, 0.4
# ======================================================
# 1) local -> world 복원 (훈련시 rotation과 정확히 반대로)
# ======================================================
def denorm_xy_batch(local_xy, origin, angle):
    # local_xy: [B,T,2]  (scaled local)
    # ✅ unscale
    local_xy = local_xy.clone()
    # local_xy[..., 0] *= SCALE_X
    # local_xy[..., 1] *= SCALE_Y

    if angle.dim() == 2:
        angle = angle.unsqueeze(-1)

    c = torch.cos(angle)
    s = torch.sin(angle)

    lx = local_xy[..., 0:1]
    ly = local_xy[..., 1:2]

    xw = lx * c - ly * s
    yw = lx * s + ly * c
    v_world = torch.cat([xw, yw], dim=-1)
    return v_world + origin

# ======================================================
# 2) 원본 CSV -> Dataset (meta 포함)
# ======================================================
class RawCSVDataset(Dataset):
    """
    원본 CSV 포맷 예시:
      frame_id, track_id, x, y, vx, vy, length, width, agent_type, ...

    여기서는 각 track_id에 대해 **딱 한 번만** (obs_len + pred_len) 윈도우를 사용.
    - track의 "처음 obs_len step"을 관측, 바로 다음 pred_len step을 예측 구간으로 사용.

    meta:
      - track_id
      - origin      : obs 마지막 위치 (world)
      - angle       : obs 첫 점 -> 마지막 점 방향 (rad)
      - obs_frames  : 관측 구간 frame_id 배열 (길이 obs_len)
      - pred_frames : 미래 구간 frame_id 배열 (길이 pred_len)
    """

    def __init__(self, csv_path, obs_len=10, pred_len=30):
        super().__init__()
        self.obs_len = obs_len
        self.pred_len = pred_len

        # 🔹 단일 CSV 파일 기준 (디렉토리 모드는 main에서 파일별로 호출)
        df = pd.read_csv(csv_path)
        df = df.sort_values(["track_id", "frame_id"])

        # track별로 모으기
        self.tracks = {}
        for tid, rows in df.groupby("track_id"):
            rows = rows.reset_index(drop=True)
            pts = rows[["x", "y"]].values.astype(np.float32)     # [N,2]
            frames = rows["frame_id"].values.astype(np.int64)    # [N]
            self.tracks[int(tid)] = {"pts": pts, "frames": frames}

        # 각 track에서 "하나의 윈도우"만 생성
        self.samples = []
        win_len = obs_len + pred_len

        for tid, data in self.tracks.items():
            pts = data["pts"]
            frs = data["frames"]
            n = len(pts)
            if n < win_len:
                continue

            i = 0  # track 처음부터 사용

            obs = pts[i: i + obs_len]                    # [obs_len,2]
            fut = pts[i + obs_len: i + win_len]          # [pred_len,2]
            obs_fr = frs[i: i + obs_len]                 # [obs_len]
            fut_fr = frs[i + obs_len: i + win_len]       # [pred_len]

            self.samples.append(
                {
                    "track_id": int(tid),
                    "obs": obs,
                    "fut": fut,
                    "obs_frames": obs_fr,
                    "pred_frames": fut_fr,
                }
            )


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        tid = s["track_id"]
        obs = s["obs"]          # [obs_len,2]
        fut = s["fut"]          # [pred_len,2]
        obs_fr = s["obs_frames"]
        fut_fr = s["pred_frames"]

        obs_len = self.obs_len
        pred_len = self.pred_len

        origin = obs[obs_len - 1:obs_len]     # [1,2]

        # translation
        ped_traj = np.concatenate([obs, fut], axis=0)
        ped_traj = ped_traj - origin

        # rotation (훈련과 동일)
        ref_point = ped_traj[0]               # translation 이후 첫 obs
        angle = np.arctan2(ref_point[1], ref_point[0])

        c = np.cos(angle)
        s = np.sin(angle)
        R = np.array([[ c, -s],
                    [ s,  c]], dtype=np.float32)

        ped_traj = ped_traj @ R
        obs_l = ped_traj[:obs_len]
        fut_l = ped_traj[obs_len:]
        ped = ped_traj
        T_total = obs_len + pred_len
        neis = ped_traj[None, :, :].astype(np.float32)  
        mask = np.ones((1, 1), dtype=np.int32)
     
        meta = {
            "track_id": np.int64(tid),
            "origin": origin.astype(np.float32),
            "angle": np.array([angle], dtype=np.float32),
            "obs_frames": obs_fr.astype(np.int64),
            "pred_frames": fut_fr.astype(np.int64),
        }

        return ped, neis, mask, meta


# ======================================================
# 3) collate_fn: batch로 합치기
# ======================================================
def coll_fn(batch):
    ped_list = []
    neis_list = []
    mask_list = []
    origin_list = []
    angle_list = []
    obs_frames_list = []
    pred_frames_list = []
    track_ids_list = []

    for ped, neis, mask, meta in batch:
        ped_list.append(ped)
        neis_list.append(neis)
        mask_list.append(mask)

        origin_list.append(meta["origin"])        # [1,2]
        angle_list.append(meta["angle"])          # [1]
        obs_frames_list.append(meta["obs_frames"])
        pred_frames_list.append(meta["pred_frames"])
        track_ids_list.append(meta["track_id"])

    ped = torch.tensor(np.stack(ped_list, axis=0), dtype=torch.float32)   # [B,T,2]
    neis = torch.tensor(np.stack(neis_list, axis=0), dtype=torch.float32) # [B,1,T,2]
    mask = torch.tensor(np.stack(mask_list, axis=0), dtype=torch.int32)   # [B,1,1]

    origins = torch.tensor(np.stack(origin_list, axis=0), dtype=torch.float32)     # [B,1,2]
    angles = torch.tensor(np.stack(angle_list, axis=0), dtype=torch.float32)       # [B,1]
    obs_frames = torch.tensor(np.stack(obs_frames_list, axis=0), dtype=torch.int64)
    pred_frames = torch.tensor(np.stack(pred_frames_list, axis=0), dtype=torch.int64)
    track_ids = torch.tensor(track_ids_list, dtype=torch.int64)

    meta_out = {
        "origin": origins,
        "angle": angles,
        "obs_frames": obs_frames,
        "pred_frames": pred_frames,
        "track_id": track_ids,
    }
    return ped, neis, mask, meta_out


# ======================================================
# 4) 모델 inference + scenario 단위 CSV 저장
# ======================================================
@torch.no_grad()
def dump_to_scenario_csv(model, loader, motion_modes, obs_len, csv_path, out_dir):
    device = next(model.parameters()).device
    model.eval()

    os.makedirs(out_dir, exist_ok=True)

    scenario_df = pd.read_csv(csv_path)

    pred_rows = []

    for ped, neis, mask, meta in loader:
        ped = ped.to(device)     # [B,T,2]
        neis = neis.to(device)   # [B,1,T,2]
        mask = mask.to(device)   # [B,1,1]

        B, T_total, _ = ped.shape
        T_pred = T_total - obs_len

        obs = ped[:, :obs_len]         # [B,obs_len,2]
        gt = ped[:, obs_len:]          # [B,T_pred,2]  (GT trajectory in local)


        neis_obs = neis[:, :, :obs_len, :]   # [B,1,obs_len,2]

        # 모델 forward
        pred_trajs, scores = model(obs, neis_obs, motion_modes, mask, None, test=True)
        # pred_trajs: [B,K,T_pred*2] or [B,K,T_pred,2]

        if pred_trajs.dim() == 3:
            Bp, K, flat = pred_trajs.shape
            assert Bp == B
            assert flat == T_pred * 2
            pred_trajs = pred_trajs.view(B, K, T_pred, 2)
        elif pred_trajs.dim() == 4:
            Bp, K, Tp, _ = pred_trajs.shape
            assert Bp == B and Tp == T_pred
        else:
            raise RuntimeError(f"Unexpected pred_trajs shape: {pred_trajs.shape}")

        # ---------- 모드별 ADE/FDE 계산 ----------
        # l2_all[b,k,t] = ||pred_{b,k,t} - gt_{b,t}||
        l2_all = torch.norm(pred_trajs - gt.unsqueeze(1), dim=-1)  # [B,K,T_pred]

        # ADE: 참고용 (원하면 logging 가능)
        ade_all = l2_all.mean(-1)            # [B,K]

        # FDE: 마지막 step의 L2 거리
        fde_all = l2_all[..., -1]            # [B,K]
        
        best_pred_score = pred_trajs[:, 0]
        
        l2 = torch.norm(pred_trajs - gt.unsqueeze(1), dim=-1)  # [B,K,T_pred]
        best_k = l2.mean(-1).argmin(-1)                        # [B]
        best_pred_minade = pred_trajs[torch.arange(B, device=device), best_k]  # [B,T_pred,2]


        # ---------- world 좌표로 복원 ----------
        origin = meta["origin"].to(device)           # [B,1,2]
        angle = meta["angle"].to(device)             # [B,1]
        pred_frames = meta["pred_frames"].to(device) # [B,T_pred]
        track_ids = meta["track_id"].to(device)      # [B]

        pred_world_score = denorm_xy_batch(best_pred_score, origin, angle)
        pred_world_minade = denorm_xy_batch(best_pred_minade, origin, angle)

        for b in range(B):
            tid = int(track_ids[b].item())
            for t in range(T_pred):
                fr = int(pred_frames[b, t].item())
                pred_rows.append((
                    tid,
                    fr,
                    pred_world_score[b, t, 0].item(),
                    pred_world_score[b, t, 1].item(),
                    # pred_world_minade[b, t, 0].item(),
                    # pred_world_minade[b, t, 1].item(),
                ))

    if len(pred_rows) == 0:
        print(f"[Warning] No predictions were generated for {csv_path}. Check dataset or lengths.")
        out_df = scenario_df.copy()
        out_df["pred_x"] = np.nan
        out_df["pred_y"] = np.nan
    else:
        pred_df = pd.DataFrame(
            pred_rows,
            columns=[
            "track_id",
            "frame_id",
            "pred_x",
            "pred_y",
            # "pred_x_minade",
            # "pred_y_minade",
        ],
        )

        out_df = scenario_df.merge(
            pred_df,
            on=["track_id", "frame_id"],
            how="left",
        )

    base_name = os.path.basename(csv_path)
    out_path = os.path.join(out_dir, base_name)

    out_df.to_csv(out_path, index=False)
# ======================================================
# 5) MAIN
# ======================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--obs_len", type=int, default=10)
    parser.add_argument("--pred_len", type=int, default=30)
    parser.add_argument("--hp_config", type=str, default="./config/ig.py")
    parser.add_argument("--csv_path", type=str,  default="/home/user/Algorithm/QCNet_cum/Code/TUTR_Contrastive/data/LIS/TEST")
    parser.add_argument("--checkpoint",type=str,default="/home/user/Algorithm/QCNet_cum/Code/TUTR_Contrastive/checkpoint/vanilla/sliding_window_10/best.pth")   
    parser.add_argument("--motion_modes_path",type=str,default="/home/user/Algorithm/QCNet_cum/Code/TUTR_Contrastive/dataset/sliding_window_10_motion_modes_200.pkl")
    parser.add_argument("--out_dir", type=str, default="/home/user/Algorithm/QCNet_cum/Code/TUTR_Contrastive/output/vanilla")
    parser.add_argument("--gpu", type=str, default="1")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[Device]", device)

    # ---------------- config (hp_config) ----------------
    spec = importlib.util.spec_from_file_location("hp_config", args.hp_config)
    hp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(hp)

    # ---------------- motion modes ---------------------
    with open(args.motion_modes_path, "rb") as f:
        mm = pickle.load(f)
    motion_modes = torch.tensor(mm, dtype=torch.float32, device=device)
    print("[motion_modes]", motion_modes.shape)

    # ---------------- model ----------------------------
    from utils.model import TrajectoryModel

    model = TrajectoryModel(
        in_size=2,
        obs_len=args.obs_len,
        pred_len=args.pred_len,
        embed_size=hp.model_hidden_dim,
        enc_num_layers=2,
        int_num_layers_list=[1, 1],
        heads=4,
        forward_expansion=2,
    ).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device)
    if "state_dict" in ckpt:
        model.load_state_dict(ckpt["state_dict"])
    else:
        model.load_state_dict(ckpt)
    print("[Model] checkpoint loaded")

    # ---------------- output dir -----------------------
    os.makedirs(args.out_dir, exist_ok=True)

    # ---------------- 파일 / 디렉토리 분기 ----------------
    if os.path.isdir(args.csv_path):
        csv_files = sorted(glob.glob(os.path.join(args.csv_path, "*.csv")))
        print(f"[INFO] Directory mode: {args.csv_path}, csv 파일 수 = {len(csv_files)}")
        if not csv_files:
            print("[WARN] 디렉토리 안에 csv 파일이 없습니다.")
            return



        for csv_file in tqdm(csv_files, desc="Processing scenarios", unit="file"):
            # print(f"[PROC] scenario file: {csv_file}")
            ds = RawCSVDataset(csv_file, obs_len=args.obs_len, pred_len=args.pred_len)
            if len(ds) == 0:
                print(f"[SKIP] {csv_file} : 유효한 (obs_len+pred_len) 윈도우가 없습니다.")
                continue

            loader = DataLoader(
                ds,
                batch_size=hp.batch_size,
                collate_fn=coll_fn,
                shuffle=False,
                num_workers=0,
                pin_memory=True,
            )

            dump_to_scenario_csv(model, loader, motion_modes, args.obs_len, csv_file, args.out_dir)

        print("[DONE] All scenario CSVs processed under directory:", args.csv_path)

    else:
        print(f"[INFO] Single file mode: {args.csv_path}")
        ds = RawCSVDataset(args.csv_path, obs_len=args.obs_len, pred_len=args.pred_len)
        if len(ds) == 0:
            print(f"[WARN] {args.csv_path} : 유효한 (obs_len+pred_len) 윈도우가 없습니다.")
        else:
            loader = DataLoader(
                ds,
                batch_size=hp.batch_size,
                collate_fn=coll_fn,
                shuffle=False,
                num_workers=0,
                pin_memory=True,
            )
            dump_to_scenario_csv(model, loader, motion_modes, args.obs_len, args.csv_path, args.out_dir)
        print("[DONE] scenario-level CSV saved under:", args.out_dir)


if __name__ == "__main__":
    main()
