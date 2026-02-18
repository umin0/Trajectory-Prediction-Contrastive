import os
import glob
import pandas as pd
import numpy as np
from itertools import combinations
from math import cos, sin, sqrt
from scipy.stats import norm
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

# =========================================================
# 1️⃣ Heading 계산 함수 (결과 컬럼명 지정)
# =========================================================
def compute_heading_from_xy(df, x_col="x", y_col="y", group_col="track_id", out_col="heading_pred"):
    """x,y 좌표 기반 heading 계산 (출력 컬럼명 지정 가능)"""
    headings = []
    for tid, group in df.groupby(group_col):
        group = group.sort_values("timestep").copy()
        x, y = group[x_col].to_numpy(), group[y_col].to_numpy()
        if len(x) < 2:
            group[out_col] = 0.0
        else:
            dx, dy = np.gradient(x), np.gradient(y)
            group[out_col] = np.arctan2(dy, dx)
        headings.append(group)
    return pd.concat(headings, ignore_index=True)


# =========================================================
# 2️⃣ CP 계산 함수 (heading 컬럼명 지정)
# =========================================================
def _calc_cp_pair(row_a, row_b, len_a, wid_a,
                  sigma_x_init, sigma_y_init, alpha_x, alpha_y, t, t_ref,
                  heading_col="heading"):
    """한 시점에서 두 차량 간 CP 계산"""
    t_rel = int(t - t_ref)
    sigma_x_t = sqrt(sigma_x_init**2 + alpha_x * t_rel)
    sigma_y_t = sqrt(sigma_y_init**2 + alpha_y * t_rel)

    dx, dy = row_b["x"] - row_a["x"], row_b["y"] - row_a["y"]
    c, s = cos(row_a[heading_col]), sin(row_a[heading_col])  # heading 컬럼 지정
    x_rel = c * dx + s * dy
    y_rel = -s * dx + c * dy

    yaw_rel = row_b[heading_col] - row_a[heading_col]
    half_LX_B = 0.5 * (len_a * abs(cos(yaw_rel)) + wid_a * abs(sin(yaw_rel)))
    half_LY_B = 0.5 * (len_a * abs(sin(yaw_rel)) + wid_a * abs(cos(yaw_rel)))
    xi, xf = -len_a/2 - half_LX_B, len_a/2 + half_LX_B
    yi, yf = -wid_a/2 - half_LY_B, wid_a/2 + half_LY_B
    px = norm.cdf(xf, x_rel, sigma_x_t) - norm.cdf(xi, x_rel, sigma_x_t)
    py = norm.cdf(yf, y_rel, sigma_y_t) - norm.cdf(yi, y_rel, sigma_y_t)
    p = px * py
    return float(np.clip(p, 0, 1))


# =========================================================
# 3️⃣ 차량쌍 CP 계산 (heading 컬럼명 지정)
# =========================================================
def compute_cp_core(df, default_len, default_wid,
                    sigma_x_init, sigma_y_init, alpha_x, alpha_y,
                    t_ref, heading_col="heading"):
    """입력 df(예측 or GT)에 대해 차량쌍 CP 계산"""
    track_groups = {tid: g for tid, g in df.groupby("track_id")}
    track_ids = list(track_groups.keys())
    results = {}

    for id_a, id_b in combinations(track_ids, 2):
        df_a, df_b = track_groups[id_a], track_groups[id_b]
        common_ts = np.intersect1d(df_a["timestep"], df_b["timestep"])
        if len(common_ts) == 0:
            continue

        cp_values = []
        for t in common_ts:
            row_a = df_a[df_a["timestep"] == t].iloc[0]
            row_b = df_b[df_b["timestep"] == t].iloc[0]

            p_ab = _calc_cp_pair(row_a, row_b, default_len, default_wid,
                                 sigma_x_init, sigma_y_init, alpha_x, alpha_y,
                                 t, t_ref, heading_col)
            p_ba = _calc_cp_pair(row_b, row_a, default_len, default_wid,
                                 sigma_x_init, sigma_y_init, alpha_x, alpha_y,
                                 t, t_ref, heading_col)
            cp_values.append(max(p_ab, p_ba))

        if cp_values:
            cp_max = float(np.max(cp_values))
            t_max = int(common_ts[np.argmax(cp_values)])
            results[(id_a, id_b)] = (cp_max, t_max)

    if not results:
        return None, None

    best_pair, (cp_max_val, cp_tmax) = max(results.items(), key=lambda kv: kv[1][0])
    return cp_max_val, cp_tmax


# =========================================================
# 4️⃣ 단일 파일 처리
# =========================================================
def compute_cp_single_file(csv_path,
                           default_len=4.7, default_wid=1.8,
                           sigma_x_init=0.5, sigma_y_init=0.5,
                           alpha_x=0.1, alpha_y=0.1):
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        return None, f"[에러] {os.path.basename(csv_path)}: {e}"

    for col in ["track_id", "timestep"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
    df = df.dropna(subset=["track_id", "timestep"]).astype({"track_id": int, "timestep": int})

    # --- Pred ---
    pred_df = df[df.get("observed", False) == False].copy() if "observed" in df.columns else df.copy()
    pred_df = pred_df.rename(columns={"pred_x": "x", "pred_y": "y"})
    pred_df = compute_heading_from_xy(pred_df, x_col="x", y_col="y", out_col="heading_pred")

    t_ref = pred_df["timestep"].min()
    pred_cp, pred_t = compute_cp_core(pred_df, default_len, default_wid,
                                      sigma_x_init, sigma_y_init, alpha_x, alpha_y,
                                      t_ref, heading_col="heading_pred")

    # --- GT ---
    gt_cp, gt_t, gt_cp_at_pred_t = None, None, None
    if {"position_x", "position_y"}.issubset(df.columns):
        gt_df = df.rename(columns={"position_x": "x", "position_y": "y"})
        # ✅ 원본 heading 그대로 사용
        gt_df = gt_df[gt_df["timestep"].between(pred_df["timestep"].min(), pred_df["timestep"].max())]

        gt_cp, gt_t = compute_cp_core(gt_df, default_len, default_wid,
                                      sigma_x_init, sigma_y_init, alpha_x, alpha_y,
                                      t_ref, heading_col="heading")  # ✅ 원본 heading 사용

    return {
        "file": os.path.basename(csv_path),
        "pred_cp_max": pred_cp,
        "pred_cp_timestep": pred_t,
        "gt_cp_max": gt_cp,
        "gt_cp_timestep": gt_t,
    }, None


# =========================================================
# 5️⃣ 폴더 단위 실행
# =========================================================
def run_folder(input_dir, out_dir,
               sigma_x_initial=0.5, sigma_y_initial=0.5,
               alpha_x=0.1, alpha_y=0.1):
    os.makedirs(out_dir, exist_ok=True)
    csv_files = sorted(glob.glob(os.path.join(input_dir, "scenario_*.csv")))
    if not csv_files:
        print(f"⚠️ {input_dir} 폴더에 CSV 없음")
        return

    all_results = []
    with ProcessPoolExecutor(max_workers=8) as ex:
        futures = [
            ex.submit(compute_cp_single_file, path,
                      4.7, 1.8,
                      sigma_x_initial, sigma_y_initial,
                      alpha_x, alpha_y)
            for path in csv_files
        ]
        for f in tqdm(as_completed(futures), total=len(futures), desc=f"Processing {os.path.basename(input_dir)}", ncols=100):
            result, err = f.result()
            if err:
                print(err)
                continue
            if result:
                all_results.append(result)

    if all_results:
        df_res = pd.DataFrame(all_results)
        df_res.to_csv(os.path.join(out_dir, "cp_max_results.csv"), index=False)
        print(f"✅ 결과 저장: {len(df_res)}개 → {out_dir}")
    else:
        print("🚫 결과 없음")



# =========================================================
# 6️⃣ 실행 예시
# =========================================================
if __name__ == "__main__":
    path_configs = [
        # (
        #     "/home/user/Algorithm/QCNet_cum/Code/Data_Processing/qcnet_output/vanilla",
        #     "/home/user/Algorithm/QCNet_cum/Code/Data_Processing/qcnet_output/vanilla/cp_max"
        # ),
        (
            "/home/user/Algorithm/QCNet_cum/Code/Data_Processing/tutr_output/vanilla",
            "/home/user/Algorithm/QCNet_cum/Code/Data_Processing/tutr_output/vanilla/cp_max"
        )
    ]

    for input_dir, out_dir in path_configs:
        run_folder(input_dir, out_dir,
                   sigma_x_initial=0.8, sigma_y_initial=0.8,
                   alpha_x=0.05, alpha_y=0.05)
