import os
import shutil
import pandas as pd
from tqdm import tqdm

# ------------------------------------------------------------
# 🔧 설정
# ------------------------------------------------------------
SUMMARY_CSV = "/home/user/Algorithm/QCNet_cum/Code/Data_Processing/tutr_output/vanilla/cp_max/cp_max_results.csv"   # summary.csv 경로
DATA_DIR = "/home/user/Algorithm/QCNet_cum/Code/Data_Processing/tutr_output/vanilla"                                    # 원본 csv들이 있는 폴더
OUTPUT_DIR = "/home/user/Algorithm/QCNet_cum/Code/Data_Processing/tutr_output/vanilla/output_cp_max"                            # 새 폴더

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ------------------------------------------------------------
# 1️⃣ Summary CSV 불러오기
# ------------------------------------------------------------
summary_df = pd.read_csv(SUMMARY_CSV)
summary_df["file"] = summary_df["file"].astype(str).str.replace(".csv", "", regex=False)

# ------------------------------------------------------------
# 2️⃣ 데이터 매칭 및 복사
# ------------------------------------------------------------
for _, row in tqdm(summary_df.iterrows(), total=len(summary_df), desc="Processing files"):
    base_name = row["file"]
    cp_val = row["pred_cp_max"]

    src_path = os.path.join(DATA_DIR, f"{base_name}.csv")
    dst_path = os.path.join(OUTPUT_DIR, f"{base_name}.csv")

    if not os.path.exists(src_path):
        print(f"⚠️ {base_name}.csv 없음, 건너뜀")
        continue

    try:
        df = pd.read_csv(src_path)

        # 🔹 cp_max 열 추가 (기존에 있으면 덮어쓰기)
        df["cp_max"] = cp_val

        # 🔹 새로운 폴더에 저장
        df.to_csv(dst_path, index=False)
    except Exception as e:
        print(f"❌ {base_name}.csv 처리 중 오류: {e}")

print(f"\n✅ 완료! 총 {len(summary_df)}개 중 {len(os.listdir(OUTPUT_DIR))}개 저장됨 → {OUTPUT_DIR}")
