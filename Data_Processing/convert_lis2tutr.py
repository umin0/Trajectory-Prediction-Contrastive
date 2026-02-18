import os
import re
import pandas as pd
from tqdm import tqdm


# =========================================
SRC_ROOT = "/home/user/Algorithm/QCNet_cum/Code/data_code/data/split"   # TRAIN, VAL, TEST가 있는 폴더
DST_ROOT = "/home/user/Algorithm/QCNet_cum/Code/data_code/data/tutr_data"  # 결과 저장 폴더
# =========================================

SPLITS = ["TRAIN", "VAL", "TEST"]


def convert_csv(in_path: str, out_path: str):
    fname = os.path.basename(in_path)

    # 파일명에서 시작 frame 추출 (_숫자.csv)
    m = re.search(r"_(\d+)\.csv$", fname)
    if not m:
        return

    frame_start = int(m.group(1))
    df = pd.read_csv(in_path)

    needed_cols = ["timestep", "track_id", "position_x",
                   "position_y", "velocity_x", "velocity_y"]

    if any(c not in df.columns for c in needed_cols):
        return

    df = df.sort_values(["timestep", "track_id"]).reset_index(drop=True)
    df["frame_id"] = frame_start + df["timestep"]

    has_cp = "cp_max" in df.columns

    df_out = pd.DataFrame({
        "frame_id": df["frame_id"].astype(int),
        "track_id": df["track_id"].astype(int),
        "x": df["position_x"].astype(float),
        "y": df["position_y"].astype(float),
        "vx": df["velocity_x"].astype(float),
        "vy": df["velocity_y"].astype(float),
    })

    if has_cp:
        df_out["cp_max"] = df["cp_max"].astype(float)

    df_out["length"] = 4.76888
    df_out["width"] = 1.87348
    df_out["agent_type"] = 0

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df_out.to_csv(out_path, index=False)


def collect_csv_files(split_root: str):
    csv_files = []
    for root, _, files in os.walk(split_root):
        for f in files:
            if f.endswith(".csv"):
                csv_files.append(os.path.join(root, f))
    return sorted(csv_files)


def check_cp_once(csv_files, split_name):
    if not csv_files:
        print(f"[{split_name}] No CSV files found.")
        return

    first_file = csv_files[0]
    df_sample = pd.read_csv(first_file, nrows=5)

    if "cp_max" in df_sample.columns:
        print(f"[{split_name}] cp_max EXISTS")
    else:
        print(f"[{split_name}] cp_max NOT FOUND")


def main():
    os.makedirs(DST_ROOT, exist_ok=True)

    total = 0

    for split in SPLITS:
        src_split = os.path.join(SRC_ROOT, split)
        if not os.path.isdir(src_split):
            print(f"[WARN] Split folder not found: {src_split}")
            continue

        dst_split = os.path.join(DST_ROOT, split)
        os.makedirs(dst_split, exist_ok=True)

        csv_files = collect_csv_files(src_split)

        # split당 한 번만 cp 존재 여부 출력
        check_cp_once(csv_files, split)

        print(f"[{split}] Converting {len(csv_files)} files...")
        total += len(csv_files)

        for in_path in tqdm(csv_files, desc=f"{split}"):
            out_path = os.path.join(dst_split, os.path.basename(in_path))
            convert_csv(in_path, out_path)

    print(f"\n🎉 Complete converting! Total processed files: {total}")


if __name__ == "__main__":
    main()
