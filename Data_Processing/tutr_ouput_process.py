# -*- coding: utf-8 -*-
"""
여러 (CSV_ROOT, TUTR_ROOT, OUT_ROOT) 쌍을
코드 내부에서 받아 TUTR pred_x/pred_y 를
원본 Argoverse2 포맷 CSV에 병합하는 스크립트
"""

import os
import re
import glob
import pandas as pd
from tqdm import tqdm


# =========================================================
# 🔹 여기서 입출력 경로 쌍을 정의
# =========================================================
IO_PAIRS = [
    {
        "csv_root": "/home/user/Algorithm/QCNet_cum/Code/Data",
        "tutr_root": "/home/user/Algorithm/QCNet_cum/Code/TUTR_Contrastive/output/vanilla",
        "out_root": "/home/user/Algorithm/QCNet_cum/Code/Data_Processing/tutr_output/vanilla",
    }
]

PRED_COL_PAIRS = [
    # ("pred_x_score", "pred_y_score"),
    # ("pred_x_minade", "pred_y_minade"),
        ("pred_x", "pred_y"),
]

# =========================================================
# TUTR index 생성
# =========================================================
def build_tutr_index(tutr_root: str):
    pattern = os.path.join(tutr_root, "**", "scenario_*.csv")
    files = glob.glob(pattern, recursive=True)

    token_to_path = {}
    for f in files:
        base = os.path.basename(f)
        m = re.match(r"scenario_(.+)\.csv$", base, flags=re.IGNORECASE)
        if not m:
            continue
        token = m.group(1)
        token_to_path[token] = f

    print(f"[INFO] TUTR csv: {len(files)} files / {len(token_to_path)} tokens")
    return token_to_path


# =========================================================
# scenario 하나 병합
# =========================================================
def merge_one_scenario(orig_csv: str, tutr_csv: str, out_csv: str):

    base_name = os.path.basename(orig_csv)
    m = re.search(r"_(\d+)\.csv$", base_name)
    if not m:
        print(f"[WARN] cannot parse frame_start: {base_name}")
        return
    frame_start = int(m.group(1))

    # -------------------------
    # 원본 CSV
    # -------------------------
    orig_df = pd.read_csv(orig_csv)
    orig_cols = list(orig_df.columns)

    if {"timestep", "track_id"} - set(orig_df.columns):
        print(f"[WARN] missing timestep/track_id: {orig_csv}")
        return

    orig_df["frame_id"] = frame_start + orig_df["timestep"].astype(int)

    # -------------------------
    # TUTR CSV
    # -------------------------
    tutr_df = pd.read_csv(tutr_csv)

    # frame_id / track_id 필수
    if {"frame_id", "track_id"} - set(tutr_df.columns):
        print(f"[WARN] missing frame_id/track_id in {tutr_csv}")
        return

    # 실제 존재하는 예측 컬럼만 선택
    pred_cols = []
    for x_col, y_col in PRED_COL_PAIRS:
        if x_col in tutr_df.columns and y_col in tutr_df.columns:
            pred_cols.extend([x_col, y_col])

    if not pred_cols:
        print(f"[WARN] no prediction columns found in {tutr_csv}")
        return

    pred_df = tutr_df[["track_id", "frame_id"] + pred_cols]

    # -------------------------
    # 병합
    # -------------------------
    merged = orig_df.merge(
        pred_df,
        on=["track_id", "frame_id"],
        how="left",
    )

    merged.drop(columns=["frame_id"], inplace=True)

    # 예측 컬럼이 없으면 NaN으로 채우기
    for c in pred_cols:
        if c not in merged.columns:
            merged[c] = pd.NA

    final_cols = orig_cols + pred_cols
    merged = merged[final_cols]

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    merged.to_csv(out_csv, index=False)



# =========================================================
# 하나의 pair 처리
# =========================================================
def process_pair(csv_root, tutr_root, out_root):

    print("\n======================================")
    print("[PROCESS PAIR]")
    print(f" CSV_ROOT : {csv_root}")
    print(f" TUTR_ROOT: {tutr_root}")
    print(f" OUT_ROOT : {out_root}")
    print("======================================")

    token_to_tutr = build_tutr_index(tutr_root)

    pattern = os.path.join(csv_root, "**", "scenario_*.csv")
    orig_csv_files = sorted(glob.glob(pattern, recursive=True))
    print(f"[INFO] original csv count: {len(orig_csv_files)}")

    for orig_csv in tqdm(orig_csv_files, desc="Merging", ncols=100):
        base = os.path.basename(orig_csv)
        m = re.match(r"scenario_(.+)\.csv$", base)
        if not m:
            continue

        token = m.group(1)
        if token not in token_to_tutr:
            continue

        out_csv = os.path.join(out_root, base)
        merge_one_scenario(orig_csv, token_to_tutr[token], out_csv)


# =========================================================
# main
# =========================================================
def main():
    for pair in IO_PAIRS:
        process_pair(
            pair["csv_root"],
            pair["tutr_root"],
            pair["out_root"],
        )

    print("\n[DONE] all pairs processed.")


if __name__ == "__main__":
    main()
