from datetime import date, timedelta
from pathlib import Path
import subprocess, sys, calendar

PY = sys.executable
SCRIPT = "preprocess_veh.py"

SITE_MAP = {
    # "36320203": "ny",
    "35310203": "ss",
}
OUTPUT_DIR = Path("./workspace")

START_Y, START_M = 2025, 7
END_Y,   END_M   = 2025, 7

def month_iter(y1, m1, y2, m2):
    y, m = y1, m1
    while (y, m) <= (y2, m2):
        yield y, m
        m += 1
        if m > 12:
            y += 1
            m = 1

def days_in_month(y, m):
    last_day = calendar.monthrange(y, m)[1]
    d = date(y, m, 1)
    for _ in range(last_day):
        yield d
        d += timedelta(days=1)

def run_one_day(site_id: str, base_name: str, d: date):
    ym = f"{d.year}{d.month:02d}"                 # YYYYMM
    ymd = d.isoformat()                           # YYYY-MM-DD
    ymd_next = (d + timedelta(days=1)).isoformat()# 다음날

    out_dir = OUTPUT_DIR / ym
    out_dir.mkdir(parents=True, exist_ok=True)

    data_name = f"{base_name}_{d.year}{d.month:02d}{d.day:02d}"   # ny_YYYYMMDD

    cmd = [
        PY, SCRIPT,
        "--source", "mongodb",
        "--site_id", site_id,
        "--start_date", ymd,         # [start, next_day)
        "--end_date",   ymd_next,    # 끝 배제 경계
        "--output_dir", str(out_dir),
        "--data_name",  data_name,
    ]
    print(">>", " ".join(cmd))
    subprocess.run(cmd, check=True)

def main():
    for y, m in month_iter(START_Y, START_M, END_Y, END_M):
        for d in days_in_month(y, m):
            for sid, base_name in SITE_MAP.items():
                run_one_day(sid, base_name, d)

if __name__ == "__main__":
    main()
