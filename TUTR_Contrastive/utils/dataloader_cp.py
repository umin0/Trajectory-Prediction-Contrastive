from typing import Optional, Sequence, List

import os, sys
import torch
import numpy as np
import pandas as pd

import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed, ThreadPoolExecutor

import pickle

class Dataloader(torch.utils.data.Dataset):

    class FixedNumberBatchSampler(torch.utils.data.sampler.BatchSampler):
        def __init__(self, n_batches, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.n_batches = n_batches
            self.sampler_iter = None #iter(self.sampler)
        def __iter__(self):
            # same with BatchSampler, but StopIteration every n batches
            counter = 0
            batch = []
            while True:
                if counter >= self.n_batches:
                    break
                if self.sampler_iter is None: 
                    self.sampler_iter = iter(self.sampler)
                try:
                    idx = next(self.sampler_iter)
                except StopIteration:
                    self.sampler_iter = None
                    if self.drop_last: batch = []
                    continue
                batch.append(idx)
                if len(batch) == self.batch_size:
                    counter += 1
                    yield batch
                    batch = []

    def __init__(self, 
        files: List[str], ob_horizon: int, pred_horizon: int,
        batch_size: int, drop_last: bool=False, shuffle: bool=False, batches_per_epoch=None, 
        frameskip: int=1, inclusive_groups: Optional[Sequence]=None,
        batch_first: bool=False, seed: Optional[int]=None,
        device: Optional[torch.device]=None,
        flip: bool=False, rotate: bool=False, scale: bool=False,
        dataset_name='sdd', dataset_type='train'
    ):
        super().__init__()
        self.ob_horizon = ob_horizon
        self.pred_horizon = pred_horizon
        self.horizon = self.ob_horizon+self.pred_horizon
        self.frameskip = int(frameskip) if frameskip and int(frameskip) > 1 else 1
        self.batch_first = batch_first
        self.flip = flip
        self.rotate = rotate
        self.scale = scale
        if device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available else "cpu") 
        else:
            self.device = device

        if inclusive_groups is None:
            inclusive_groups = [[] for _ in range(len(files))]
        assert(len(inclusive_groups) == len(files))

        print(" Scanning files...")
        files_ = []
        for path, incl_g in zip(files, inclusive_groups):
            if os.path.isdir(path):
                files_.extend([(os.path.join(root, f), incl_g) \
                    for root, _, fs in os.walk(path) \
                    for f in fs if f.endswith(".csv")])
            elif os.path.exists(path):
                files_.append((path, incl_g))
        data_files = sorted(files_, key=lambda _: _[0])

        data = []
        
        done = 0
        # too large of max_workers will cause the problem of memory usage
        max_workers = min(len(data_files), torch.get_num_threads(), 16)
        with ProcessPoolExecutor(mp_context=multiprocessing.get_context("spawn"), max_workers=max_workers) as p:
            futures = [p.submit(self.__class__.load, self, f, incl_g) for f, incl_g in data_files]
            for fut in as_completed(futures):
                done += 1
                sys.stdout.write("\r\033[K Loading data files...{}/{}".format(
                    done, len(data_files)
                ))
            for fut in futures:
                item = fut.result()
                if item is not None:
                    data.extend(item)
                sys.stdout.write("\r\033[K Loading data files...{}/{} ".format(
                    done, len(data_files)
                ))
        self.data = np.array(data, dtype=object)
        print("\n   {} trajectories loaded.".format(len(self.data)))
        save_file = './dataset/'
        if not os.path.exists(save_file):
            os.mkdir(save_file)
        save_path_file = save_file + dataset_name + '_' + dataset_type +  '.pkl'
        f = open(save_path_file, 'wb')
        pickle.dump(data, f)
        f.close()
        del data
        self.rng = np.random.RandomState()
        if seed: self.rng.seed(seed)

        if shuffle:
            sampler = torch.utils.data.sampler.RandomSampler(self)
        else:
            sampler = torch.utils.data.sampler.SequentialSampler(self)
        if batches_per_epoch is None:
            self.batch_sampler = torch.utils.data.sampler.BatchSampler(sampler, batch_size, drop_last)
            self.batches_per_epoch = len(self.batch_sampler)
        else:
            self.batch_sampler = self.__class__.FixedNumberBatchSampler(batches_per_epoch, sampler, batch_size, drop_last)
            self.batches_per_epoch = batches_per_epoch

    def collate_fn(self, batch):
        X, Y, NEIGHBOR, CP_MAX = [], [], [], []
        for item in batch:
            if len(item) == 4:
                hist, future, neighbor, cp_val = item
            else:
                hist, future, neighbor = item
                cp_val = 0.0   # or None

            X.append(hist)
            Y.append(future)
            NEIGHBOR.append(neighbor)
            CP_MAX.append(cp_val)

            hist_shape = hist.shape     # [L_ob, D]
            neighbor_shape = neighbor.shape  # [1, D] or [L, N, D]

            # copy to avoid modifying original data
            hist_aug = hist.copy()
            future_aug = future.copy()
            neighbor_aug = neighbor.copy()

            # augmentation
            if self.flip:
                if self.rng.randint(2):
                    hist_aug[..., 1] *= -1  # flip y
                    future_aug[..., 1] *= -1
                    neighbor_aug[..., 1] *= -1
                if self.rng.randint(2):
                    hist_aug[..., 0] *= -1  # flip x
                    future_aug[..., 0] *= -1
                    neighbor_aug[..., 0] *= -1

            if self.rotate:
                rot = self.rng.random() * (2 * np.pi)
                s, c = np.sin(rot), np.cos(rot)
                R = np.array([[c, -s], [s, c]])
                # rotate position, velocity, acceleration (first 6 dims)
                if hist_aug.shape[-1] >= 2:
                    hist_aug[..., :2] = (R @ hist_aug[..., :2].T).T
                if hist_aug.shape[-1] >= 4:
                    hist_aug[..., 2:4] = (R @ hist_aug[..., 2:4].T).T
                if hist_aug.shape[-1] >= 6:
                    hist_aug[..., 4:6] = (R @ hist_aug[..., 4:6].T).T
                # apply to future and neighbor too
                if future_aug.shape[-1] >= 2:
                    future_aug[..., :2] = (R @ future_aug[..., :2].T).T
                if neighbor_aug.shape[-1] >= 2:
                    neighbor_aug[..., :2] = (R @ neighbor_aug[..., :2].T).T
                if neighbor_aug.shape[-1] >= 4:
                    neighbor_aug[..., 2:4] = (R @ neighbor_aug[..., 2:4].T).T
                if neighbor_aug.shape[-1] >= 6:
                    neighbor_aug[..., 4:6] = (R @ neighbor_aug[..., 4:6].T).T

            if self.scale:
                s = self.rng.randn() * 0.05 + 1.0  # N(1,0.05)
                hist_aug[..., :6] *= s
                future_aug[..., :2] *= s
                neighbor_aug[..., :6] *= s

            X.append(hist_aug)
            Y.append(future_aug)
            NEIGHBOR.append(neighbor_aug)

        # pad neighbors
        n_neighbors = [n.shape[1] if n.ndim == 3 else 1 for n in NEIGHBOR]
        max_neighbors = max(n_neighbors)
        data_dim = X[0].shape[-1]

        if max_neighbors != min(n_neighbors):
            NEIGHBOR = [
                np.pad(nei, ((0, 0), (0, max_neighbors - n), (0, 0)),
                    "constant", constant_values=1e9)
                if nei.ndim == 3 else
                np.pad(nei[np.newaxis, ...], ((0, 0), (0, max_neighbors - 1), (0, 0)),
                    "constant", constant_values=1e9)
                for nei, n in zip(NEIGHBOR, n_neighbors)
            ]

        # stacking
        stack_dim = 0 if self.batch_first else 1
        x = np.stack(X, stack_dim)
        y = np.stack(Y, stack_dim)
        neighbor = np.stack(NEIGHBOR, stack_dim)

        x = torch.tensor(x, dtype=torch.float32, device=self.device)
        y = torch.tensor(y, dtype=torch.float32, device=self.device)
        neighbor = torch.tensor(neighbor, dtype=torch.float32, device=self.device)
        cp_max = torch.tensor(CP_MAX, dtype=torch.float32, device=self.device)
        return x, y, neighbor, cp_max



    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

    @staticmethod
    def load(self, filename, inclusive_groups):
        if os.path.isdir(filename):
            return None

        df = pd.read_csv(filename)
        required_cols = ['frame_id', 'track_id', 'x', 'y']
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Missing required column {col} in {filename}")

        has_v = {'vx', 'vy'}.issubset(df.columns)
        has_a = {'ax', 'ay'}.issubset(df.columns)
        has_cp = 'cp_max' in df.columns   # ✅ 추가: cp_max 여부 확인

        data = {}
        for _, row in df.iterrows():
            t = int(row['frame_id'])
            idx = int(row['track_id'])
            entry = [float(row['x']), float(row['y'])]
            if has_v:
                entry += [float(row['vx']), float(row['vy'])]
            if has_a:
                entry += [float(row['ax']), float(row['ay'])]
            if 'agent_type' in df.columns:
                entry += [float(row['agent_type'])]
            if has_cp:
                entry += [float(row['cp_max'])]   # ✅ cp_max 값 추가
            if t not in data:
                data[t] = {}
            data[t][idx] = entry

        # 기존 velocity, acceleration 계산
        data = self.extend(data, self.frameskip, compute_v=not has_v, compute_a=not has_a)

        # horizon 처리
        time = np.sort(list(data.keys()))
        if len(time) < (self.horizon - 1) * self.frameskip + 1:
            return None

        traj = []
        e = len(time)
        tid0 = 0
        while tid0 < e - (self.horizon - 1) * self.frameskip:
            tid1 = tid0 + (self.horizon - 1) * self.frameskip
            t0 = time[tid0]
            agents = list(data[t0].keys())
            for i in agents:
                hist = []
                future = []
                for step in range(0, self.horizon):
                    t = time[tid0 + step * self.frameskip]
                    if i in data[t]:
                        hist.append(data[t][i])
                    else:
                        break
                if len(hist) == self.horizon:
                    hist_arr = np.array(hist[:self.ob_horizon])
                    future_arr = np.array(hist[self.ob_horizon:], dtype=np.float32)
                    neighbor = np.full((self.horizon, 1, len(hist_arr[0])), 1e9, dtype=np.float32)
                    cp_val = float(hist_arr[0, -1]) if has_cp else None  # ✅ cp_max가 있으면 마지막 열 사용
                    traj.append((hist_arr, future_arr[:, :2], neighbor, cp_val))

            tid0 += 1

        items = []
        for hist, future, neighbor, cp_val in traj:
            item = (np.float32(hist), np.float32(future), np.float32(neighbor))
            if cp_val is not None:
                item = (np.float32(hist), np.float32(future), np.float32(neighbor), cp_val)  # ✅ cp_max 포함
            items.append(item)
        return items

    
    def extend(self, data, frameskip, compute_v=True, compute_a=True):
        time = np.sort(list(data.keys()))
        if len(time) < 2:
            return data
        dt = np.min(np.diff(time))

        # Velocity
        if compute_v:
            for tid in range(len(time) - frameskip):
                t0, t1 = time[tid], time[tid + frameskip]
                for i in list(data[t1].keys()):
                    if i in data[t0]:
                        x0, y0 = data[t0][i][0:2]
                        x1, y1 = data[t1][i][0:2]
                        vx, vy = (x1 - x0) / dt, (y1 - y0) / dt
                        data[t1][i][2:2] = [vx, vy]
                        if tid == 0 or i not in data[time[tid - 1]]:
                            data[t0][i][2:2] = [vx, vy]

        # Acceleration
        if compute_a:
            for tid in range(len(time) - frameskip):
                t0, t1 = time[tid], time[tid + frameskip]
                for i in list(data[t1].keys()):
                    if i in data[t0] and len(data[t0][i]) >= 4:
                        vx0, vy0 = data[t0][i][2:4]
                        vx1, vy1 = data[t1][i][2:4]
                        ax, ay = (vx1 - vx0) / dt, (vy1 - vy0) / dt
                        data[t1][i][4:4] = [ax, ay]
                        if tid == 0 or i not in data[time[tid - 1]]:
                            data[t0][i][4:4] = [ax, ay]

        return data


if __name__ == '__main__':

    dataset_list = ['']

    
