# LIS data extraction for QCNet training

## version
- 2025.08.01 ver 1.0

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
    --json_dir /workspace/aimmo_data_extraction/data_sample \
    --output_dir /workspace/aimmo_data_extraction/data_sample \
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
    --output_dir /workspace/aimmo_data_extraction/data_extracted \
    --data_name test
```

## 3. 출력 파일

1. `{data_name}_object.csv`: 객체 정보
   - ObjectID, SourceID, EquipmentType, ObjectType, VehicleWidth, VehicleLength, StartDate

2. `{data_name}_track.csv`: 트래킹 정보
   - FrameCount, ObjectID, VehicleClass, DistanceX, DistanceY, Speed, Heading, AccelerationX, AccelerationY