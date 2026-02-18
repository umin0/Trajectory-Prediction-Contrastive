import os
import importlib
import torch

from utils.dataloader import Dataloader
from utils.utils import seed, get_rng_state

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--data_root", type=str, required=True,
                    help="Root path containing TRAIN/VAL/TEST folders")
parser.add_argument("--frameskip", type=int, default=1)
parser.add_argument("--config", type=str, required=True)
parser.add_argument("--device", type=str, default=None)
parser.add_argument("--seed", type=int, default=1)

# 예)
# python get_data_pkl.py --data_root data/eth --config config/eth.py
# data/eth/TRAIN, data/eth/VAL, data/eth/TEST 구조


def pick_dataset_name(data_root: str) -> str:
    """
    dataset_name을 기존 코드처럼 str(path).split('/')[1]에 의존하지 않고,
    data_root의 마지막 폴더명을 사용하도록 안전하게 처리.
    """
    return os.path.basename(os.path.normpath(data_root))


def make_inclusive_list(config, n_paths: int):
    if getattr(config, "INCLUSIVE_GROUPS", None) is not None:
        return [config.INCLUSIVE_GROUPS for _ in range(n_paths)]
    return None


if __name__ == "__main__":
    settings = parser.parse_args()

    spec = importlib.util.spec_from_file_location("config", settings.config)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)

    if settings.device is None:
        settings.device = "cuda" if torch.cuda.is_available() else "cpu"
    settings.device = torch.device(settings.device)

    seed(settings.seed)
    init_rng_state = get_rng_state(settings.device)
    rng_state = init_rng_state

    kwargs = dict(
        batch_first=False,
        frameskip=settings.frameskip,
        ob_horizon=config.OB_HORIZON,
        pred_horizon=config.PRED_HORIZON,
        device=settings.device,
        seed=settings.seed
    )

    data_root = settings.data_root
    dataset_name = pick_dataset_name(data_root)

    # split 폴더 경로 자동 구성
    split_paths = {
        "train": [os.path.join(data_root, "TRAIN")],
        "val":   [os.path.join(data_root, "VAL")],
        "test":  [os.path.join(data_root, "TEST")],
    }

    # 존재하지 않으면 skip
    for k in list(split_paths.keys()):
        if not os.path.isdir(split_paths[k][0]):
            print(f"[WARN] Split folder not found, skip: {split_paths[k][0]}")
            split_paths[k] = []

    # =========================
    # TRAIN
    # =========================
    if split_paths["train"]:
        print("[TRAIN]", split_paths["train"])
        inclusive = make_inclusive_list(config, len(split_paths["train"]))
        train_dataset = Dataloader(
            split_paths["train"],
            **kwargs,
            inclusive_groups=inclusive,
            flip=True, rotate=True, scale=True,
            batch_size=config.batch_size,
            shuffle=True,
            batches_per_epoch=config.EPOCH_BATCHES,
            dataset_type="train",
            dataset_name=dataset_name
        )

    # =========================
    # VAL (no augmentation)
    # =========================
    if split_paths["val"]:
        print("[VAL]", split_paths["val"])
        inclusive = make_inclusive_list(config, len(split_paths["val"]))
        val_dataset = Dataloader(
            split_paths["val"],
            **kwargs,
            inclusive_groups=inclusive,
            flip=False, rotate=False, scale=False,
            batch_size=config.batch_size,
            shuffle=False,
            dataset_type="val",
            dataset_name=dataset_name
        )

    # =========================
    # TEST
    # =========================
    if split_paths["test"]:
        print("[TEST]", split_paths["test"])
        inclusive = make_inclusive_list(config, len(split_paths["test"]))
        test_dataset = Dataloader(
            split_paths["test"],
            **kwargs,
            inclusive_groups=inclusive,
            batch_size=config.batch_size,
            shuffle=False,
            dataset_type="test",
            dataset_name=dataset_name
        )
