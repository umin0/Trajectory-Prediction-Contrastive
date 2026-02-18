import os
import random
import shutil
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed


def split_dataset(
    src_root: str,
    dst_root: str,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int = 42,
    num_workers: int = 8,
):
    # ✅ 비율 검증
    if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-9:
        raise ValueError("train_ratio + val_ratio + test_ratio must sum to 1.0")

    random.seed(seed)

    # TRAIN / VAL / TEST 자동 생성
    train_root = os.path.join(dst_root, "TRAIN")
    val_root   = os.path.join(dst_root, "VAL")
    test_root  = os.path.join(dst_root, "TEST")

    os.makedirs(train_root, exist_ok=True)
    os.makedirs(val_root, exist_ok=True)
    os.makedirs(test_root, exist_ok=True)

    # 1️⃣ *_av2 폴더 수집
    target_month_folders = [
        d for d in os.listdir(src_root)
        if d.endswith("_av2") and os.path.isdir(os.path.join(src_root, d))
    ]

    # 2️⃣ 모든 시나리오 폴더 수집
    all_folders = []
    for month_folder in target_month_folders:
        month_path = os.path.join(src_root, month_folder)
        subfolders = [
            os.path.join(month_path, sub)
            for sub in os.listdir(month_path)
            if os.path.isdir(os.path.join(month_path, sub))
        ]
        all_folders.extend(subfolders)

    print(f"📁 전체 시나리오 개수: {len(all_folders)}")

    if not all_folders:
        print("⚠️ 데이터가 없습니다.")
        return

    # 3️⃣ 셔플
    random.shuffle(all_folders)
    n = len(all_folders)

    n_train = int(n * train_ratio)
    n_val   = int(n * val_ratio)
    n_test  = n - n_train - n_val

    train_folders = all_folders[:n_train]
    val_folders   = all_folders[n_train:n_train + n_val]
    test_folders  = all_folders[n_train + n_val:]

    print(f"📦 TRAIN: {len(train_folders)} ({train_ratio*100:.1f}%)")
    print(f"📦 VAL:   {len(val_folders)} ({val_ratio*100:.1f}%)")
    print(f"📦 TEST:  {len(test_folders)} ({test_ratio*100:.1f}%)")

    # 4️⃣ 병렬 복사
    def copy_folder(src_folder, dst_split_root):
        dst_path = os.path.join(dst_split_root, os.path.basename(src_folder))
        shutil.copytree(src_folder, dst_path, dirs_exist_ok=True)

    def parallel_copy(samples, dst_split_root, label):
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(copy_folder, f, dst_split_root) for f in samples]
            for _ in tqdm(as_completed(futures), total=len(futures), desc=f"Copying {label}"):
                pass

    parallel_copy(train_folders, train_root, "TRAIN")
    parallel_copy(val_folders,   val_root,   "VAL")
    parallel_copy(test_folders,  test_root,  "TEST")

    print("\n✅ 데이터 분할 완료!")
    print(f"저장 위치: {dst_root}")
    print(" ├── TRAIN")
    print(" ├── VAL")
    print(" └── TEST")


# =====================================================
# 🔧 여기서 경로 + 비율 설정
# =====================================================
if __name__ == "__main__":

    # 📂 입력 폴더
    src_root = "/home/user/Algorithm/QCNet_cum/Code/data_code/data"

    # 📂 출력 루트 (자동으로 TRAIN/VAL/TEST 생성됨)
    dst_root = "/home/user/Algorithm/QCNet_cum/Code/data_code/data/split"

    # 📊 비율 설정 (여기서만 바꾸면 됨)
    train_ratio = 0.4
    val_ratio   = 0.1
    test_ratio  = 0.5

    split_dataset(
        src_root=src_root,
        dst_root=dst_root,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=42,
        num_workers=8,
    )



