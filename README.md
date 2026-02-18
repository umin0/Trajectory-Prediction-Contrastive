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


# LIS data extraction for QCNet training


## 설치

### 1. 환경 설정
- 패키지 설치
```bash
pip install -r requirements.txt
```
or
```bash
pip install pandas numpy pymongo tqdm python-dateutil python-dotenv pyarrow dnspython
```

- (mongoDB 접속 시) 환경변수 설정
1. .env 파일 설정
```
# MongoDB Connection Settings
MONGO_URI={접속 uri}
MONGO_DATABASE=infra
MONGO_COLLECTION=log
```
### 2. 폴더 구조
```
aimmo_data_extraction/   
├── .env                    # MongoDB 연결 설정   
├── requirements.txt        # 패키지 목록   
├── preprocess_{class}.py      # class = veh, ped / 클래스 별 추출 스크립트   
└── README.md              
```

## 사용법

### JSON 파일에서 데이터 처리

```bash
python preprocess_veh.py \
    --source json \
    --json_dir /path/to/json/files \
    --output_dir /path/to/output \
    --data_name my_dataset
```

### 예시
```bash
python preprocess_veh.py \
    --source json \
    --json_dir /data_sample \
    --output_dir /data_sample \
    --data_name test
```

## 2. MongoDB에서 쿼리를 통한 데이터 처리

### 기본 사용법

```bash
python preprocess_veh.py \
    --source mongodb \
    --site_id "35310203" \
    --start_date 2025-01-01 \
    --end_date 2025-01-02 \
    --output_dir /data_extracted \
    --data_name test
```

## 3. 출력 파일

1. `{data_name}_object.csv`: 객체 정보
   - ObjectID, SourceID, EquipmentType, ObjectType, VehicleWidth, VehicleLength, StartDate

2. `{data_name}_track.csv`: 트래킹 정보
   - FrameCount, ObjectID, VehicleClass, DistanceX, DistanceY, Speed, Heading, AccelerationX, AccelerationY

## 4. QCNet 학습용 변환

- lis2av2/script.sh 스크립트 활용
