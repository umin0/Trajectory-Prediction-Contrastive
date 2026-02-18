import os, re, glob, math
import pandas as pd
import numpy as np
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from typing import Dict, Any, List, Tuple

# ========== 기본 설정 ==========
PRED_START_TS = 10
PRED_LEN = 30

# 정규식 precompile (속도 향상)
RE_AGENT = re.compile(r"^#\s*Agent\s+ID:\s*(\d+)\s*$", re.MULTILINE)
RE_MODE_PROBS = re.compile(r"#\s*Mode\s*Probabilities\s*\(pi\)\s*:\s*\[([0-9eE\+\-.,\s]+)\]", re.MULTILINE)
RE_MODE = re.compile(r"^##\s*Mode\s*(\d+)\s*\(Prob:\s*([0-9.+\-eE]+)\)\s*$", re.MULTILINE)
RE_POINT = re.compile(r"(?:t\s*=\s*)?(\d+)\s*,\s*x\s*=\s*([\-0-9\.eE]+)\s*,\s*y\s*=\s*([\-0-9\.eE]+)")

# ==========================================================
def parse_prediction_txt(txt_path: str) -> Dict[str, List[Tuple[np.ndarray, np.ndarray, float]]]:
    """TXT 내 Agent별 mode 궤적 및 확률 파싱"""
    # 'utf-8-sig'를 사용하여 BOM(Byte Order Mark) 문제를 처리하여 안정성을 높입니다.
    with open(txt_path, encoding="utf-8-sig") as f:
        text = f.read()

    preds = {}
    agents = list(RE_AGENT.finditer(text))

    for i, am in enumerate(agents):
        aid = am.group(1) # Agent ID는 문자열로 유지
        start = am.end()
        end = agents[i + 1].start() if i + 1 < len(agents) else len(text)
        block = text[start:end]

        # Mode 확률 벡터 추출
        probs_match = RE_MODE_PROBS.search(block)
        mode_probs = []
        if probs_match:
            try:
                mode_probs = [float(x.strip()) for x in probs_match.group(1).split(",") if x.strip() != ""]
            except ValueError:
                # 확률 파싱 오류 시 경고
                print(f"경고: {txt_path}에서 Mode 확률 파싱 오류.", file=sys.stderr)
                mode_probs = []

        mode_headers = list(RE_MODE.finditer(block))
        modes = []
        for j, mh in enumerate(mode_headers):
            mstart = mh.end()
            mend = mode_headers[j + 1].start() if j + 1 < len(mode_headers) else len(block)
            body = block[mstart:mend]

            xs, ys = np.full(PRED_LEN, np.nan), np.full(PRED_LEN, np.nan)
            for pm in RE_POINT.finditer(body):
                t = int(pm.group(1))
                if 0 <= t < PRED_LEN:
                    try:
                        xs[t] = float(pm.group(2))
                        ys[t] = float(pm.group(3))
                    except ValueError:
                        # 좌표 파싱 오류 시 경고
                        print(f"경고: {txt_path}에서 좌표 파싱 오류.", file=sys.stderr)
                        continue

            # 확률은 mode_probs 벡터에서 가져옴
            prob = mode_probs[j] if j < len(mode_probs) else 0.0
            # 최소한 첫 번째 점이라도 있어야 유효한 모드로 간주
            if not np.all(np.isnan(xs)) or not np.all(np.isnan(ys)):
                modes.append((xs, ys, prob))
                
        if modes:
            preds[aid] = modes
    return preds


# ==========================================================
def choose_highest_prob_mode(modes: List[Tuple[np.ndarray, np.ndarray, float]]) -> int:
    """가장 확률 높은 모드 선택"""
    probs = np.array([m[2] for m in modes])
    return np.argmax(probs)


# ==========================================================
def fill_predictions_fast(csv_path: str, preds: Dict[str, Any], out_path: str) -> None:
    """TXT 기반 예측 결과를 CSV에 삽입하고, 쓰기 안정성을 확보합니다."""
    df = pd.read_csv(csv_path)
    if "pred_x" not in df.columns:
        df["pred_x"] = np.nan
        df["pred_y"] = np.nan

    arr_x, arr_y = df["position_x"].to_numpy(), df["position_y"].to_numpy()
    arr_ts, arr_tid = df["timestep"].to_numpy(), df["track_id"].to_numpy()
    pred_x, pred_y = df["pred_x"].to_numpy(), df["pred_y"].to_numpy()

    tids = np.unique(arr_tid)
    for tid in tids:
        # Agent ID는 정규식으로 인해 문자열로 파싱되므로 비교를 위해 문자열로 변환
        tid_str = str(tid)
        if tid_str not in preds:
            continue
        
        idxs = np.where(arr_tid == tid)[0]
        # 원본 코드에는 last_idx가 있었지만 사용되지 않음. 삭제하거나 주석 처리.
        # last_idx = idxs[arr_ts[idxs].argmax()] 
        modes = preds[tid_str]

        # ✅ 확률이 가장 높은 모드 선택
        best_mode_idx = choose_highest_prob_mode(modes)
        xs, ys, _ = modes[best_mode_idx]

        mask = (arr_tid == tid) & (arr_ts >= PRED_START_TS) & (arr_ts < PRED_START_TS + PRED_LEN)
        ts_sorted = np.argsort(arr_ts[mask])
        sel_idx = np.where(mask)[0][ts_sorted]
        L = min(len(sel_idx), PRED_LEN)
        
        if L > 0:
            pred_x[sel_idx[:L]] = xs[:L]
            pred_y[sel_idx[:L]] = ys[:L]

    df["pred_x"], df["pred_y"] = pred_x, pred_y
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    
    # 1. CSV 파일로 저장
    df.to_csv(out_path, index=False)
    
    # 2. 💡 (핵심 개선) OS의 쓰기 버퍼를 강제로 디스크에 플러시하여 파일 유실 방지
    try:
        # 파일을 다시 'a' (append) 모드로 열어 파일 기술자를 얻습니다.
        with open(out_path, 'a') as f:
            f.flush()
            # os.fsync()는 OS 레벨에서 디스크 쓰기 완료를 요청합니다 (I/O 안정성 확보).
            os.fsync(f.fileno()) 
            
    except Exception as e:
        # I/O 에러 발생 시 예외를 다시 던져서 메인 프로세스에 보고합니다.
        raise IOError(f"파일 저장 후 강제 플러싱 실패 ({out_path}): {e}")


# ==========================================================
def process_one(csv_path: str, txt_dir: str, out_dir: str) -> None:
    """단일 CSV 파일을 처리하고 해당 TXT 파일의 예측 결과를 삽입합니다."""
    # 시나리오 ID 추출
    scen_id_match = re.search(r"scenario_([\w\-]+)\.csv$", os.path.basename(csv_path))
    if not scen_id_match:
        return
        
    scen_id = scen_id_match.group(1)
    txt_path = os.path.join(txt_dir, f"{scen_id}.txt")
    out_path = os.path.join(out_dir, os.path.basename(csv_path))
    
    # 1. TXT 파일 존재 여부 확인
    if not os.path.exists(txt_path):
        # TXT 파일이 없는 경우는 정상적인 건너뛰기로 처리
        print(f"\n[SKIP] TXT 파일 없음 (ID: {scen_id})", file=sys.stderr)
        return
        
    # 2. 예측 결과 파싱
    preds = parse_prediction_txt(txt_path)
    if not preds:
        print(f"\n[SKIP] 예측 결과 없음/파싱 실패 (ID: {scen_id})", file=sys.stderr)
        return
        
    # 3. CSV에 예측 결과 삽입 및 안정적 저장
    fill_predictions_fast(csv_path, preds, out_path)
    
    # 4. 최종 확인 및 로그
    # if os.path.exists(out_path):
    #     # 성공 시 최종 경로 명시적으로 출력
    #     print(f"\n[SUCCESS] 파일 저장 완료: {os.path.basename(out_path)}", file=sys.stderr)
    # else:
    #     # 파일이 존재하지 않으면 I/O 실패로 간주하고 예외 발생
    #     raise IOError(f"파일 저장 실패: {out_path} 경로에 파일이 생성되지 않았습니다.")


# ==========================================================
def process_scenario_data(csv_dir: str, txt_dir: str, out_dir: str, max_workers: int = 6) -> None:
    """여러 시나리오 파일을 병렬로 처리합니다."""
    # 모든 CSV 경로 재귀적으로 탐색
    csv_paths = sorted(glob.glob(os.path.join(csv_dir, "**", "scenario_*.csv"), recursive=True))
    if not csv_paths:
        print(f"경고: {csv_dir}에서 처리할 CSV 파일을 찾을 수 없습니다.", file=sys.stderr)
        return
        
    os.makedirs(out_dir, exist_ok=True)
    
    successful_count = 0
    failed_count = 0
    
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        # future 객체와 해당 파일 경로를 매핑하여 저장
        futures = {ex.submit(process_one, path, txt_dir, out_dir): path for path in csv_paths}
        
        # as_completed를 사용하여 작업 완료를 추적
        for future in tqdm(as_completed(futures), total=len(futures), desc=f"{os.path.basename(csv_dir)} 병렬처리", ncols=100):
            csv_path = futures[future]
            try:
                # future.result()를 호출하여 결과를 얻거나 예외를 발생시킵니다.
                future.result() 
                successful_count += 1
            except Exception as e:
                failed_count += 1
                # ❌ 실패 시 해당 파일과 오류 메시지를 출력
                print(f"\n❌ Error processing {os.path.basename(csv_path)}: {e}", file=sys.stderr)
                
    print(f"\n=======================================================")
    print(f"✅ 최종 완료: {out_dir}")
    print(f"   성공적으로 처리된 파일: {successful_count}개")
    print(f"   처리 실패한 파일: {failed_count}개")
    if failed_count > 0:
        print("💡 위 출력된 '❌ Error processing' 메시지를 확인하여 실패한 파일의 원인을 점검하세요.")
    print(f"=======================================================")


# ==========================================================
if __name__ == "__main__":
    # Windows 환경이므로 경로에 Raw String (r"...") 사용을 권장합니다.
    CONFIGS = [
        {
             "CSV_DIR": "/home/user/Algorithm/QCNet_cum/Code/Data",
            "TXT_DIR": r"/home/user/Algorithm/QCNet_cum/Code/QCNet_Contrasive/output",
            "OUT_DIR": "/home/user/Algorithm/QCNet_cum/Code/Data_Processing/qcnet_output/vanilla",
        }            
    ]
    for conf in CONFIGS:
        process_scenario_data(conf["CSV_DIR"], conf["TXT_DIR"], conf["OUT_DIR"])