# Trajectory Prediction Contrastive Learning

***Vehicle Intelligence and Control Lab, Ajou University***

---

This codebase implements the system described in the paper:

**Query-centric trajectory prediction**

Zikang Zhou, Jianping Wang, et al. Published in *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2023*.

See the [paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Zhou_Query-Centric_Trajectory_Prediction_CVPR_2023_paper.pdf) for more details.

**Trajectory Unified Transformer for Pedestrian Trajectory Prediction**

Lei Shi, Lijun Wang, Shuo Zhou, and Gang Hua. Published in *Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2023*.

See the [paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Shi_Trajectory_Unified_Transformer_for_Pedestrian_Trajectory_Prediction_ICCV_2023_paper.pdf) for more details.

Please contact **Bongsob Song** ([bsong@ajou.ac.kr](mailto:bsong@ajou.ac.kr)) if you have any questions.


# 코드 실행 순서

##### STEP 1. Data_Extraction
##### STEP 2. Data_Processing - Input
##### STEP 3. QCNet_Contrasive/TUTR_Contrastive
##### STEP 4. Data_Processing - Output
##### STEP 5. Evaluation - Collision Probability

## STEP 1. Data_Extraction

### 1. 환경 설정
- 패키지 설치
```bash
pip install -r requirements.txt
```

- mongoDB 접속 시 .env 파일 환경변수 설정
```
# MongoDB Connection Settings
MONGO_URI={접속 uri}
MONGO_DATABASE=infra
MONGO_COLLECTION=log
```
### 2. 실행

- 데이터 다운로드를 위한 설정
- LIS SITE MAP 설정 
```bash
SITE_MAP = {
    # "36320203": "ny", #남양읍
    "35310203": "ss", #새솔동
}
```
- 출력 경로 설정 
```bash
OUTPUT_DIR = Path("./workspace")
```
- 날짜 설정 
```bash
START_Y, START_M = 2025, 7
END_Y,   END_M   = 2025, 7
```
- 터미널을 통한 다운로드 실행 명령어
```bash
python download.py
```

### 3. 출력 파일

1. `{data_name}_object.csv`: 객체 정보
   - ObjectID, SourceID, EquipmentType, ObjectType, VehicleWidth, VehicleLength, StartDate

2. `{data_name}_track.csv`: 트래킹 정보
   - FrameCount, ObjectID, VehicleClass, DistanceX, DistanceY, Speed, Heading, AccelerationX, AccelerationY
  

## STEP 2. Data_Processing - Input

### 1. Sliding Window 기반 데이터 구성

- 입력 / 출력 경로 설정
```bash
ROOT_DIR = "/home/user/Algorithm/QCNet_cum/Code/aimmo_data_extraction/workspace"
OUT_ROOT = "/home/user/Algorithm/QCNet_cum/Code/data_code/data"
```

- 슬라이딩 윈도우 설정
```bash
INPUT_FRAME  = 10     # 관측 길이
OUTPUT_FRAME = 30     # 예측 길이
WINDOW_STEP  = 10     # 슬라이딩 간격
MIN_AGENTS   = 2      # 최소 차량 수
```

- 정지 차량 제거 임계값 설정
```bash
STATIONARY_THRESH = 0.05
```

- 터미널을 통한 변환 실행 명령어
```bash
python preprocessing_lis_sliding.py
```

### 2. 데이터셋 분할 (TRAIN / VAL / TEST)

- 입력 / 출력 경로 설정
```bash
src_root = "/home/user/Algorithm/QCNet_cum/Code/data_code/data"
dst_root = "/home/user/Algorithm/QCNet_cum/Code/data_code/data/split"
```
- 분할 비율 설정
```bash
train_ratio = 0.4
val_ratio   = 0.1
test_ratio  = 0.5
```
- 터미널을 통한 데이터 분할 실행 명령어
```bash
python selecting_train_val_test_data.py
```

- 출력 폴더 구조
 ```bash
split/
 ├── TRAIN/
 ├── VAL/
 └── TEST/
```

### 3. 학습데이터 포맷 변환 
#### 3-1. QCNet 학습용 포맷 변환
- 입력 / 출력 경로 설정
```bash
source_root = "/home/user/Algorithm/QCNet_cum/Code/data_code/data/split"
output_root = "/home/user/Algorithm/QCNet_cum/Code/data_code/data/qcnet_data"
json_source_file = "/home/user/Algorithm/QCNet_cum/Code/data_code/SS_map.json"
```
- 터미널을 통한 변환 실행 명령어
```bash
python convert_lis2argoverse2.py
```
- 출력 구조
```bash
qcnet_data/
 ├── train/
 │    └── raw/
 │         └── <scenario_id>/
 │              ├── scenario_<scenario_id>.parquet
 │              └── log_map_archive_<scenario_id>.json
 ├── val/
 │    └── raw/
 │         └── <scenario_id>/
 │              ├── scenario_<scenario_id>.parquet
 │              └── log_map_archive_<scenario_id>.json
 └── test/
      └── raw/
           └── <scenario_id>/
                ├── scenario_<scenario_id>.parquet
                └── log_map_archive_<scenario_id>.json
```
#### 3-2. TUTR 학습용 포맷 변환
- 입력 / 출력 경로 설정
```bash
SRC_ROOT = "/home/user/Algorithm/QCNet_cum/Code/data_code/data/split"
DST_ROOT = "/home/user/Algorithm/QCNet_cum/Code/data_code/data/tutr_data"
```
- 터미널을 통한 변환 실행 명령어
```bash
python convert_lis2tutr.py
```
- 출력 폴더 구조
```bash
tutr_data/
 ├── TRAIN/
 │    └── scenario_<scenario_id>.csv
 ├── VAL/
 │    └── scenario_<scenario_id>.csv
 └── TEST/
      └── scenario_<scenario_id>.csv
```

## STEP 3. QCNet_Contrasive
### 1. 모델 학습
#### 1-1. Vanilla 학습
- 파라미터 설정
 ```bash
    args = parser.parse_args([
        '--root',                   '/home/user/Algorithm/QCNet_cum/Code/QCNet_Contrasive/Data/LIS', # 데이터 경로 설정
        '--train_batch_size',       '16', # 배치 사이즈 설정
        '--val_batch_size',         '16', # 배치 사이즈 설정
        '--test_batch_size',        '16', # 배치 사이즈 설정
        '--devices',                '6',  # 디바이스(GPU) 설정
        '--dataset',                'argoverse_v2_ACL',
        '--num_historical_steps',   '10', # 입력 시퀀스 길이 설정
        '--num_future_steps',       '30', # 출력 시퀀스 길이 설정
        # 공간 반경(Interaction Radius) 설정(원문 논문과 동일)
        '--num_recurrent_steps',    '3', 
        '--pl2pl_radius',           '150',
        '--time_span',              '10',
        '--pl2a_radius',            '50',
        '--a2a_radius',             '50',
        '--num_t2m_steps',          '30',
        '--pl2m_radius',            '150',
        '--a2m_radius',             '150'
    ])
  ```
- 학습 실행 명령어
```bash
python train_qcnet.py
```
- Note 1: 처음으로 학습 스크립트를 실행할 경우, 데이터 전처리 과정에 몇 시간이 소요될 수 있습니다.
- Note 2: 학습 중 생성되는 체크포인트는 자동으로 lightning_logs/ 폴더에 저장됩니다.

#### 1-2. Contrastive learning 학습
- 파라미터 설정
 ```bash
    args = parser.parse_args([
        '--root',                   '/home/user/Algorithm/QCNet_cum/Code/QCNet_Contrasive/Data/LIS', # 데이터 경로 설정
        '--train_batch_size',       '16', # 배치 사이즈 설정
        '--val_batch_size',         '16', # 배치 사이즈 설정
        '--test_batch_size',        '16', # 배치 사이즈 설정
        '--devices',                '6',  # 디바이스(GPU) 설정
        '--dataset',                'argoverse_v2_ACL',
        '--num_historical_steps',   '10', # 입력 시퀀스 길이 설정
        '--num_future_steps',       '30', # 출력 시퀀스 길이 설정
        # 공간 반경(Interaction Radius) 설정(원문 논문과 동일)
        '--num_recurrent_steps',    '3', 
        '--pl2pl_radius',           '150',
        '--time_span',              '10',
        '--pl2a_radius',            '50',
        '--a2a_radius',             '50',
        '--num_t2m_steps',          '30',
        '--pl2m_radius',            '150',
        '--a2m_radius',             '150'
        # Contrastive learning 파라미터
        '--lambda_contrastive',     '1',
        '--tau',                    '0.5',
        '--theta_p',                '0.33',
        '--theta_n',                '0.50',
    ])
  ```
- Contrastive Learning 학습 실행 명령어
```bash
python train_qcnet_contrastive.py
```
OR
- CP기반 Contrastive Learning 학습할 경우, 
```bash
python train_qcnet_contrastive_cp.py
```

- Note 1: 처음으로 학습 스크립트를 실행할 경우, 데이터 전처리 과정에 몇 시간이 소요될 수 있습니다.
- Note 2: 학습 중 생성되는 체크포인트는 자동으로 lightning_logs/ 폴더에 저장됩니다.

### 2. 모델 평가  

- 저장된 체크포인트(.ckpt)를 로드해서 TEST split에서 trainer.validate()로 성능을 평가합니다.
- 파라미터 설정
 ```bash
 args = parser.parse_args([
        '--model',      'QCNet', 
        '--root', '/home/user/Algorithm/QCNet_cum/Code/QCNet_Contrasive/Data',  # 데이터 경로 설정
        '--ckpt_path','/home/user/Algorithm/QCNet_cum/Code/QCNet_Contrasive/lightning_logs/Vanilla/checkpoints/epoch=70-step=7100.ckpt' # 체크포인트 경로 설정
    ])
 ```
- 평가 실행 명령어
```bash
python val_qcnet.py
```
### 3. 모델 추론 및 예측 결과 저장  

- 저장된 체크포인트(.ckpt)를 로드하여 TEST 데이터에 대해 추론을 수행합니다.
- 모델의 예측 결과는 --save_dir 경로에 저장됩니다.
- 파라미터 설정
 ```bash
 args = parser.parse_args([
        '--model',      'QCNet', 
        '--root', '/home/user/Algorithm/QCNet_cum/Code/QCNet_Contrasive/Data',  # 데이터 경로 설정
        '--ckpt_path','/home/user/Algorithm/QCNet_cum/Code/QCNet_Contrasive/lightning_logs/Vanilla/checkpoints/epoch=70-step=7100.ckpt' # 체크포인트 경로 설정
        '--save_dir', '/home/user/Algorithm/QCNet_cum/Code/QCNet_Contrasive/output' # 추론 결과 저장 경로 설정
    ])
 ```
- 추론 실행 명령어
```bash
python test_qcnet.py
```

## STEP 3. TUTR_Contrasive
### 1. 입력 데이터 전처리
- 파라미터 설정
```bash
parser.add_argument("--data_root", type=str, default='./data/LIS') # 데이터 경로 설정
parser.add_argument("--config", type=str, default='config/ig.py') # Config 파일 지정
```
- 전처리 실행 명령어
```bash
python get_data_pkl.py
```

### 2. 모델 학습
#### 2-1. Vanilla 학습
- 파라미터 설정
```bash
parser.add_argument('--dataset_path', type=str, default='./dataset/') # 데이터 경로 설정
parser.add_argument('--dataset_name', type=str, default='LIS') # 데이터명 설정
parser.add_argument("--hp_config", type=str, default='config/ig.py', help='hyper-parameter') # Config 파일 지정
parser.add_argument('--lr_scaling', action='store_true', default=False)
parser.add_argument('--num_works', type=int, default=2)
parser.add_argument('--obs_len', type=int, default=10) # 입력 시퀀스 길이 설정
parser.add_argument('--pred_len', type=int, default=30) # 출력 시퀀스 길이 설정
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--gpu', type=str, default='0') # 디바이스(GPU) 설정
parser.add_argument('--data_scaling', type=list, default=[1.9, 0.4])
parser.add_argument('--checkpoint', type=str, default='./checkpoint/vanilla') # 체크포인트 저장 경로 설정
```
- 학습 실행 명령어
```bash
python train_tutr.py
```

#### 2-2. Contrastive Learning 학습

- config 파일 파라미터 설정(config/ig_contrastive.py)
```bash
lambda_contr = 0.5      # λ
contr_tau = 0.5      # τ
contr_pos_delta = 0.33   # θ_p
contr_neg_delta = 0.33   # θ_n
```
- 파라미터 설정
```bash
parser.add_argument('--dataset_path', type=str, default='./dataset/') # 데이터 경로 설정
parser.add_argument('--dataset_name', type=str, default='LIS') # 데이터명 설정
parser.add_argument("--hp_config", type=str, default='config/ig_contrastive.py', help='hyper-parameter') # Config 파일 지정
parser.add_argument('--obs_len', type=int, default=10) # 입력 시퀀스 길이 설정
parser.add_argument('--pred_len', type=int, default=30) # 출력 시퀀스 길이 설정
parser.add_argument('--gpu', type=str, default='0') # 디바이스(GPU) 설정
parser.add_argument('--checkpoint', type=str, default='./checkpoint/vanilla') # 체크포인트 저장 경로 설정
```

- Contrastive Learning 학습 실행 명령어
```bash
python train_tutr_contrastive.py
```
OR
- CP기반 Contrastive Learning 학습할 경우, 
```bash
python train_tutr_contrastive_cp.py
```


### 3. 모델 평가
- 저장된 체크포인트(.pth)를 로드해서 TEST 데이터에 대해 성능을 평가합니다.
- 파라미터 설정
```bash
parser.add_argument('--dataset_path', type=str, default='./dataset/') # 데이터 경로 설정
parser.add_argument('--dataset_name', type=str, default='sliding_window_10') # 데이터명 설정
parser.add_argument("--hp_config", type=str, default="./config/ig.py") # Config 파일 지정
parser.add_argument("--checkpoint",type=str,default="/home/user/Algorithm/QCNet_cum/Code/TUTR_Contrastive/checkpoint/vanilla") # 체크포인트 경로 설정
parser.add_argument('--obs_len', type=int, default=10) # 입력 시퀀스 길이 설정
parser.add_argument('--pred_len', type=int, default=30) # 출력 시퀀스 길이 설정
```

- 평가 실행 명령어
```bash
python val_tutr.py
```

### 4. 모델 추론 및 예측 결과 저장  
- 저장된 체크포인트(.pth)를 로드하여 TEST 데이터에 대해 추론을 수행합니다.
- 모델의 예측 결과는 --out_dir 경로에 저장됩니다.
- 파라미터 설정
```bash
parser.add_argument("--obs_len", type=int, default=10) # 입력 시퀀스 길이 설정
parser.add_argument("--pred_len", type=int, default=30) # 출력 시퀀스 길이 설정
parser.add_argument("--hp_config", type=str, default="./config/ig.py") # Config 파일 지정
parser.add_argument("--csv_path", type=str,  default="/home/user/Algorithm/QCNet_cum/Code/TUTR_Contrastive/data/LIS/TEST") # 테스트 데이터 경로 설정
parser.add_argument("--checkpoint",type=str,default="/home/user/Algorithm/QCNet_cum/Code/TUTR_Contrastive/checkpoint/vanilla/sliding_window_10/best.pth") # 체크포인트 경로 설정
parser.add_argument("--motion_modes_path",type=str,default="/home/user/Algorithm/QCNet_cum/Code/TUTR_Contrastive/dataset/sliding_window_10_motion_modes_200.pkl") # 모션모드 경로 설정
parser.add_argument("--out_dir", type=str, default="/home/user/Algorithm/QCNet_cum/Code/TUTR_Contrastive/output/vanilla") # 추론 결과 저장 경로 설정
 ```
- 추론 실행 명령어
```bash
python test_tutr.py
```

## STEP 4. Data_Processing - Output
### 1. QCNet 추론 결과 처리
- 모델이 저장한 *.txt 예측 결과를 파싱하여 원본 scenario_*.csv에 pred_x, pred_y 컬럼을 추가합니다.
- 각 agent(track_id)별로 가장 확률이 높은 모드(Highest-Prob Mode) 를 선택해 예측 궤적을 삽입합니다.
- 결과 CSV는 OUT_DIR에 저장됩니다.
- 경로 설정
```bash
CONFIGS = [
        {   "CSV_DIR": "/home/user/Algorithm/QCNet_cum/Code/Data", #입력 CSV 경로 설정
            "TXT_DIR": r"/home/user/Algorithm/QCNet_cum/Code/QCNet_Contrasive/output", # 입력 TXT 경로 설정 (QCNet모델이 저장한 예측 결과 txt)
            "OUT_DIR": "/home/user/Algorithm/QCNet_cum/Code/Data_Processing/qcnet_output/vanilla", # 출력 경로 설정
        }]
```
- 실행 명령어
```bash
python qcnet_ouput_process.py
```

### 2. TUTR 추론 결과 처리
- TUTR에서 생성된 예측 결과(pred_x/pred_y)를 Argoverse2 포맷 scenario_*.csv에 병합합니다.
- track_id + frame_id 기준으로 join하여 예측 컬럼을 원본 CSV에 추가합니다.
- 결과는 out_root에 동일한 파일명으로 저장됩니다.
- 경로 설정
```bash
CONFIGS = [
    {   "csv_root": "/home/user/Algorithm/QCNet_cum/Code/Data",  #입력 CSV 경로 설정
        "tutr_root": "/home/user/Algorithm/QCNet_cum/Code/TUTR_Contrastive/output/vanilla", # 입력 csv 경로 설정 (TUTR모델이 저장한 예측 결과 csv)
        "out_root": "/home/user/Algorithm/QCNet_cum/Code/Data_Processing/tutr_output/vanilla", # 출력 경로 설정
    }]
```
- 실행 명령어
```bash
python tutr_ouput_process.py
```

## STEP 5. Evaluation - Collision Probability
### Collision Probability (CP) 계산
- 각 scenario_*.csv 파일에 대해
  - 예측 궤적(pred_x, pred_y) 기반 CP
  - GT 궤적(position_x, position_y) 기반 CP
- 차량 쌍 간 CP를 계산하고
- scenario별 최대 CP (CP_max) 값을 추출합니다.
- 경로 설정
```bash
path_configs = [
        (   "/home/user/Algorithm/QCNet_cum/Code/Data_Processing/tutr_output/vanilla", # 입력 데이터 경로
            "/home/user/Algorithm/QCNet_cum/Code/Data_Processing/tutr_output/vanilla/cp_max"  # 출력 파일 저장 경로
        )]
```
- 파라미터 설정
```bash
sigma_x_initial=0.8,
sigma_y_initial=0.8,
alpha_x=0.05,
alpha_y=0.05
```
- cp max 계산 실행 명령어
```bash
python calculate_cp_max.py
```

### Collision Probability (CP) 평가
- 전체 데이터에서 gt_cp_max의 최대값을 구하고 그 값보다 더 큰 pred_cp_max가 몇 개인지 계산합니다.
- 모델이 GT에서 발생한 최대 위험보다 더 과도하게 위험을 예측한 비율을 측정합니다.
- 경로 설정
```bash
csv_path = "K:/final/TEST_OUTPUT/QCNet/contrastive/cp_final_gt/cp_results_separated_heading_prob.csv"
```
- cp max 평가 실행 명령어
```bash
python eval_cp_max.py
```
