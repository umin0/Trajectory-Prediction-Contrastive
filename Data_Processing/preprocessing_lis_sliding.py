import os
import re
import numpy as np
import pandas as pd
from tqdm import tqdm

# =====================================================
ROOT_DIR = "/home/user/Algorithm/QCNet_cum/Code/aimmo_data_extraction/workspace"                 # 입력 루트: Origin/202409, Origin/202410, ...
OUT_ROOT = "/home/user/Algorithm/QCNet_cum/Code/data_code/data"     # 최종 출력 루트
INPUT_FRAME = 10                      # 관측 길이
OUTPUT_FRAME = 30                     # 예측 길이
WINDOW_STEP = 10                      # 슬라이딩 홉(프레임)
MIN_AGENTS = 2                        # 저장할 최소 차량 수(윈도우 내 track_id 개수)

STATIONARY_THRESH = 0.05              # 정지차량 판정 임계값 (mean|vx|, mean|vy|)

# 파일명에서 날짜(YYYYMMDD) 추출 
FILENAME_PATTERN = re.compile(r"(\d{8})_track\.csv$", re.IGNORECASE)


# =====================================================
# 1) heading+speed 기반 vx/vy 계산
# =====================================================
def compute_velocity(df: pd.DataFrame, heading_deg_col="Heading", speed_col="Speed"):
    hdg_deg = df[heading_deg_col].astype(float)
    hdg_rad = np.deg2rad(hdg_deg)
    speed = df[speed_col].astype(float)
    vx = speed * np.cos(hdg_rad)
    vy = speed * np.sin(hdg_rad)
    return hdg_deg, vx, vy


def stationary_track_ids(df: pd.DataFrame, thr=STATIONARY_THRESH):
    hdg_deg, vx, vy = compute_velocity(df, "Heading", "Speed")
    tmp = pd.DataFrame({
        "trackId": df["ObjectID"].astype(int).values,
        "abs_vx": np.abs(vx),
        "abs_vy": np.abs(vy),
    })
    g = tmp.groupby("trackId", as_index=False).agg(
        mean_abs_vx=("abs_vx", "mean"),
        mean_abs_vy=("abs_vy", "mean"),
    )
    ids = g[(g["mean_abs_vx"] <= thr) & (g["mean_abs_vy"] <= thr)]["trackId"].tolist()
    return ids


# =====================================================
# 2)  원본 *_track.csv → 표준 스키마(out df)로 변환 + 정지차량 제거
#    출력 df 컬럼: trackId, frame, xCenter, yCenter, heading(deg), xVelocity, yVelocity, ...
# =====================================================
def preprocess_one_track_csv(track_csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(track_csv_path)
    df.sort_values(by=["ObjectID", "FrameCount"], inplace=True)

    required = ["ObjectID", "FrameCount", "DistanceX", "DistanceY", "Heading", "Speed"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # (1) 정지차량 trackId 찾고 제거
    stop_ids = stationary_track_ids(df, STATIONARY_THRESH)
    if stop_ids:
        df = df[~df["ObjectID"].isin(stop_ids)].copy()

    if len(df) == 0:
        return pd.DataFrame()  # 빈 데이터

    # (2) 남은 데이터로 vx/vy 재계산
    hdg_deg, vx, vy = compute_velocity(df, "Heading", "Speed")
    xAcc = df["AccelerationX"].astype(float) if "AccelerationX" in df.columns else 0.0
    yAcc = df["AccelerationY"].astype(float) if "AccelerationY" in df.columns else 0.0

    out = pd.DataFrame({
        "trackId": df["ObjectID"].astype(int).values,
        "frame": df["FrameCount"].astype(int).values,
        "xCenter": df["DistanceX"].astype(float).values,
        "yCenter": df["DistanceY"].astype(float).values,
        "heading": hdg_deg,   # deg 유지 (둘째 단계에서 rad 변환)
        "xVelocity": vx,
        "yVelocity": vy,
        "xAcceleration": xAcc,
        "yAcceleration": yAcc,
    })

    # cp_max가 원본에 있으면 "track 기준 상수" 또는 "row별"일 수 있으니 그대로 보존
    if "cp_max" in df.columns:
        out["cp_max"] = df["cp_max"].astype(float).values

    # Maneuver가 있으면 보존
    if "Maneuver" in df.columns:
        out["Maneuver"] = df["Maneuver"].values

    return out


# =====================================================
# 3) 표준 스키마 df → AV2/TUTR 스타일 컬럼명으로 변경 + 메타 컬럼 추가
# =====================================================
def to_av2_like_schema(df_std: pd.DataFrame, scenario_date: str, data_num: str, frame_interval: int) -> pd.DataFrame:
    # 컬럼명 매핑
    column_mapping = {
        "trackId": "track_id",
        "frame": "timestep",
        "xCenter": "position_x",
        "yCenter": "position_y",
        "heading": "heading",
        "xVelocity": "velocity_x",
        "yVelocity": "velocity_y",
        "Maneuver": "Maneuver",
        "cp_max": "cp_max",
    }
    df = df_std.rename(columns={k: v for k, v in column_mapping.items() if k in df_std.columns}).copy()

    # scenario_id / Data_Num
    df["scenario_id"] = scenario_date
    df["Data_Num"] = data_num

    # 메타 컬럼 기본값
    for col in ["observed", "object_type", "object_category", "start_timestamp",
                "end_timestamp", "num_timestamps", "focal_track_id", "city"]:
        if col not in df.columns:
            df[col] = ""

    df["object_category"] = 0
    df["focal_track_id"] = 0
    df["num_timestamps"] = frame_interval
    df["start_timestamp"] = 0
    df["end_timestamp"] = 0
    df["city"] = "kcity"
    df["object_type"] = "VEHICLE"

    # 출력 컬럼 순서
    base_cols = [
        "observed", "track_id", "object_type", "object_category", "timestep",
        "position_x", "position_y", "heading", "velocity_x", "velocity_y",
        "scenario_id", "start_timestamp", "end_timestamp", "num_timestamps",
        "focal_track_id", "city", "Data_Num"
    ]
    df = df[base_cols]
    return df


# =====================================================
# 4) 슬라이딩 윈도우로 scenario 생성 + 저장
# =====================================================
def sliding_and_save(df_av2: pd.DataFrame, out_folder: str, date_info: str,
                     input_frame: int, output_frame: int, window_step: int, min_agents: int):
    frame_interval = input_frame + output_frame
    max_frame = int(df_av2["timestep"].max())

    # start_frame: 0, step, 2*step ...
    for start_frame in range(0, max_frame - frame_interval + 2, window_step):
        end_frame = start_frame + frame_interval
        df_group = df_av2[(df_av2["timestep"] >= start_frame) & (df_av2["timestep"] < end_frame)].copy()
        if df_group.empty:
            continue

        # 로컬 타임스텝 + observed + heading rad
        df_group.loc[:, "timestep"] = df_group["timestep"] - start_frame
        df_group.loc[:, "observed"] = (df_group["timestep"] <= (input_frame - 1))
        df_group.loc[:, "heading"] = np.radians(df_group["heading"].astype(float))

        # 시나리오 id 재설정
        new_scenario_id = f"{date_info}_{start_frame}"
        df_group["scenario_id"] = str(new_scenario_id)

        # window 안에서 모든 프레임을 가진 track만 유지
        df_group = df_group.groupby("track_id").filter(
            lambda x: len(x["timestep"].unique()) == frame_interval
        )
        if df_group.empty:
            continue

        # 평균 속도가 완전 0인 정지 트랙 제거
        df_group = df_group[
            (df_group.groupby("track_id")["velocity_x"].transform("mean").abs() > STATIONARY_THRESH ) |
            (df_group.groupby("track_id")["velocity_y"].transform("mean").abs() > STATIONARY_THRESH )
        ]
        if df_group.empty:
            continue

        # 최소 차량 수 조건
        n_agents = df_group["track_id"].nunique()
        if n_agents < min_agents:
            continue
        
        df_group.loc[:, "object_category"] = 3

        # 저장
        new_file_name = f"scenario_{new_scenario_id}.csv"
        new_folder_path = os.path.join(out_folder, f"{new_scenario_id}")
        os.makedirs(new_folder_path, exist_ok=True)
        new_file_path = os.path.join(new_folder_path, new_file_name)
        df_group.to_csv(new_file_path, index=False)


# =====================================================
# 5) 메인: ROOT_DIR 하위 월폴더 순회 → 각 *_track.csv 처리 → scenario 생성
# =====================================================
def main():
    if not os.path.isdir(ROOT_DIR):
        raise FileNotFoundError(f"ROOT_DIR not found: {ROOT_DIR}")

    os.makedirs(OUT_ROOT, exist_ok=True)

    subdirs = sorted([d for d in os.listdir(ROOT_DIR) if os.path.isdir(os.path.join(ROOT_DIR, d))])
    total_files = 0

    frame_interval = INPUT_FRAME + OUTPUT_FRAME

    for sub in subdirs:
        sub_path = os.path.join(ROOT_DIR, sub)
        track_files = sorted([f for f in os.listdir(sub_path) if f.endswith("_track.csv")])

        if not track_files:
            print(f"[SKIP] No *_track.csv in {sub_path}")
            continue

        # 최종 출력: OUT_ROOT/<월폴더>_av2
        out_month_dir = os.path.join(OUT_ROOT, f"{sub}_av2")
        os.makedirs(out_month_dir, exist_ok=True)

        print(f"\n[INFO] Processing folder {sub} ({len(track_files)} files)")
        for file_index, fname in enumerate(tqdm(track_files, desc=f"{sub}", unit="file")):
            fpath = os.path.join(sub_path, fname)

            # 파일명에서 날짜 뽑기(없으면 UNKNOWN)
            m = FILENAME_PATTERN.search(fname)
            date_str = m.group(1) if m else f"UNKNOWN_{file_index:04}"
            data_num = f"{file_index:04}"

            try:
                # (1) 정지차량 제거 + 표준 스키마 변환
                df_std = preprocess_one_track_csv(fpath)
                if df_std.empty:
                    continue

                # (2) AV2-like 컬럼 정리 + 메타 추가
                df_av2 = to_av2_like_schema(df_std, scenario_date=date_str, data_num=data_num, frame_interval=frame_interval)

                # (3) 슬라이딩 윈도우 시나리오 생성 + 저장
                sliding_and_save(
                    df_av2, out_month_dir, date_str,
                    input_frame=INPUT_FRAME, output_frame=OUTPUT_FRAME,
                    window_step=WINDOW_STEP, min_agents=MIN_AGENTS
                )

                total_files += 1

            except Exception as e:
                print(f"[ERROR] {fpath}: {e}")
                continue

    print(f"\n[Done] Processed {total_files} files under {ROOT_DIR}.")
    print(f"[Done] Results saved in {OUT_ROOT}/<subdir>_av2/<scenario_id>/scenario_<scenario_id>.csv")


if __name__ == "__main__":
    main()
