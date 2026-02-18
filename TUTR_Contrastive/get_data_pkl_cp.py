import importlib
import torch
import os

from utils.dataloader_cp import Dataloader
from utils.utils import seed, get_rng_state

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--train", nargs='+', default=[])
parser.add_argument("--test", nargs='+', default=[])
parser.add_argument("--frameskip", type=int, default=1)
parser.add_argument("--config", type=str, default=None)
parser.add_argument("--device", type=str, default=None)
parser.add_argument("--seed", type=int, default=1)

# python get_data_pkl_cpmax.py --train data/lis/train_bench_contrastive_for_cp --config config/ig_bench_baseline.py

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

    ###############################################################################
    #####                                                                    ######
    ##### prepare datasets                                                   ######
    #####                                                                    ######
    ###############################################################################
    kwargs = dict(
        batch_first=False,
        frameskip=settings.frameskip,
        ob_horizon=config.OB_HORIZON,
        pred_horizon=config.PRED_HORIZON,
        device=settings.device,
        seed=settings.seed
    )

    train_data, test_data = None, None

    # ==========================================================
    # ✅ TRAIN SET 처리 (cp_max 포함)
    # ==========================================================
    if settings.train:
        print(f"[INFO] Train datasets: {settings.train}")

        if config.INCLUSIVE_GROUPS is not None:
            inclusive = [config.INCLUSIVE_GROUPS for _ in range(len(settings.train))]
        else:
            inclusive = None

        # Dataloader 내부에서 자동으로 pkl 파일 저장
        train_dataset = Dataloader(
            settings.train,
            **kwargs,
            inclusive_groups=inclusive,
            flip=True, rotate=True, scale=True,
            batch_size=config.batch_size,
            shuffle=True,
            batches_per_epoch=config.EPOCH_BATCHES,
            dataset_type='train',
            dataset_name=str(settings.train[0]).split('/')[1]
            # extra_columns=['cp_max']  # ✅ <== cp_max 포함하도록 추가
        )

    # ==========================================================
    # ✅ TEST SET 처리 (cp_max 포함)
    # ==========================================================
    if settings.test:
        print(f"[INFO] Test datasets: {settings.test}")

        if config.INCLUSIVE_GROUPS is not None:
            inclusive = [config.INCLUSIVE_GROUPS for _ in range(len(settings.test))]
        else:
            inclusive = None

        test_dataset = Dataloader(
            settings.test,
            **kwargs,
            inclusive_groups=inclusive,
            batch_size=config.batch_size,
            shuffle=False,
            dataset_type='test',
            dataset_name=str(settings.test[0]).split('/')[1],
            extra_columns=['cp_max']  # ✅ test에도 cp_max 반영
        )

    print("\n✅ All done! PKL files saved with cp_max field included.")
