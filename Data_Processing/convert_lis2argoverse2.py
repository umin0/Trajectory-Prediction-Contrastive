import os
import shutil
import glob
import re
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

SCENARIO_ID_PATTERN = re.compile(r"scenario_([\w\-]+)\.csv$", re.IGNORECASE)


# ===============================================================
# 단일 CSV 처리
# ===============================================================
def convert_one(csv_path, output_root_dir, json_source_path, suffix, use_custom_filename):
    file_name = os.path.basename(csv_path)
    m = SCENARIO_ID_PATTERN.search(file_name)
    if not m:
        return

    scenario_id = m.group(1)
    scenario_id_with_suffix = f"{suffix}{scenario_id}" if suffix else scenario_id

    # 출력 폴더 생성
    output_subfolder = os.path.join(output_root_dir, scenario_id_with_suffix)
    os.makedirs(output_subfolder, exist_ok=True)

    parquet_name = (
        f"scenario_{scenario_id_with_suffix}.parquet"
        if use_custom_filename else file_name.replace(".csv", ".parquet")
    )
    parquet_path = os.path.join(output_subfolder, parquet_name)

    try:
        df = pd.read_csv(csv_path, engine="pyarrow")
        df.to_parquet(parquet_path, index=False, engine="pyarrow")
    except Exception as e:
        print(f"[❌ CSV Error] {file_name}: {e}")
        return

    # JSON 복사
    json_new_name = f"log_map_archive_{scenario_id_with_suffix}.json"
    json_dest_path = os.path.join(output_subfolder, json_new_name)
    try:
        shutil.copyfile(json_source_path, json_dest_path)
    except Exception as e:
        print(f"[❌ JSON Error] {file_name}: {e}")


# ===============================================================
# 폴더 하나 처리 (TRAIN 또는 VAL 또는 TEST)
# ===============================================================
def process_split_folder(source_dir, output_root_dir, json_source_path,
                         suffix=None, use_custom_filename=True, num_workers=8):

    if not os.path.exists(source_dir):
        print(f"⚠️ Skip (not found): {source_dir}")
        return

    os.makedirs(output_root_dir, exist_ok=True)

    csv_paths = sorted(
        glob.glob(os.path.join(source_dir, "**", "scenario_*.csv"), recursive=True)
    )

    if not csv_paths:
        print(f"⚠️ '{source_dir}'에서 CSV를 찾지 못했습니다.")
        return

    print(f"\n📂 {os.path.basename(source_dir)} → {len(csv_paths)}개 처리 시작")

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(
                convert_one, csv, output_root_dir,
                json_source_path, suffix, use_custom_filename
            )
            for csv in csv_paths
        ]

        for _ in tqdm(as_completed(futures), total=len(futures),
                      desc=f"Processing {os.path.basename(source_dir)}"):
            pass


# ===============================================================
# TRAIN / VAL / TEST 자동 처리
# ===============================================================
def convert_dataset_structure(
    source_root,
    output_root,
    json_source_path,
    suffix=None,
    use_custom_filename=True,
    num_workers=8
):

    split_map = {
        "TRAIN": os.path.join(output_root, "train", "raw"),
        "VAL":   os.path.join(output_root, "val", "raw"),
        "TEST":  os.path.join(output_root, "test", "raw"),
    }

    for split_name, output_path in split_map.items():
        source_path = os.path.join(source_root, split_name)
        process_split_folder(
            source_dir=source_path,
            output_root_dir=output_path,
            json_source_path=json_source_path,
            suffix=suffix,
            use_custom_filename=use_custom_filename,
            num_workers=num_workers
        )

    print("\n🎉 TRAIN / VAL / TEST 전체 변환 완료!")


# ===============================================================
# 🔧 사용 예시
# ===============================================================
if __name__ == "__main__":

    source_root = "/home/user/Algorithm/QCNet_cum/Code/data_code/data/split"
    output_root = "/home/user/Algorithm/QCNet_cum/Code/data_code/data/qcnet_data"
    json_source_file = "/home/user/Algorithm/QCNet_cum/Code/data_code/SS_map.json"

    convert_dataset_structure(
        source_root=source_root,
        output_root=output_root,
        json_source_path=json_source_file,
        suffix=None,
        use_custom_filename=True,
        num_workers=8
    )



