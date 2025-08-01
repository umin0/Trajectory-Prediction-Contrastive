import json
import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import os
import glob
from tqdm import tqdm
import pymongo
from datetime import datetime, timedelta
from dotenv import load_dotenv

"""
----- Entire process -----
1. load json files from directory OR load from MongoDB
2. basic preprocessing (scale, rotation, speed calculation)
3. filtering (size-based, short objects)
4. save csv file of object information
5. save csv file of track information
"""

# MongoDB 설정 (고정값)
MONGO_DATABASE = "infra"
MONGO_COLLECTION = "log"
load_dotenv()  # .env 파일 로드
MONGO_URI = os.getenv("MONGO_URI")  # 환경변수에서 가져오기

def load_from_mongodb(connection_string, start_date=None, end_date=None, site_id=None):
    """
    MongoDB에서 차량 데이터를 불러와서 DataFrame으로 변환
    
    Parameters:
    - connection_string: MongoDB 연결 문자열
    - start_date: 시작 날짜 (timestamp in milliseconds)
    - end_date: 종료 날짜 (timestamp in milliseconds)
    - site_id: 특정 사이트 ID (선택사항)
    
    Returns:
    - DataFrame: MongoDB에서 불러온 데이터프레임
    """
    # MongoDB 연결
    client = pymongo.MongoClient(connection_string)
    db = client[MONGO_DATABASE]
    collection = db[MONGO_COLLECTION]
    
    # 쿼리 조건 설정
    query = {}
    
    if start_date and end_date:
        query['sDSMTimeStamp'] = {
            '$gte': start_date,
            '$lte': end_date
        }
    elif start_date:
        query['sDSMTimeStamp'] = {'$gte': start_date}
    elif end_date:
        query['sDSMTimeStamp'] = {'$lte': end_date}
    
    # 차량만 필터링하는 쿼리 (이미 MongoDB에서 필터링)
    query['DetectedObjectList'] = {
        '$elemMatch': {
            'objType': 1
        }
    }
    
    if site_id:
        query['Site_ID'] = int(site_id)
    
    print(f"MongoDB 쿼리 조건: {query}")
    print(f"Database: {MONGO_DATABASE}, Collection: {MONGO_COLLECTION}")
    
    # 데이터 불러오기
    print("MongoDB에서 데이터를 불러오는 중...")
    cursor = collection.find(query)
    
    df_list = []
    for msg in tqdm(cursor, desc="Loading from MongoDB"):
        site_id = msg['Site_ID']
        date_time = msg['datetime']
        msg_count = msg['msgCnt']
        timestamp = msg['sDSMTimeStamp']
        ref_pos = msg['refPos']
        
        for obj in msg['DetectedObjectList']:
            obj_id = obj['objectID']
            obj_type = obj['objType']
            if obj_type != 1:  # 차량만 선택
                continue
                
            heading = obj['heading']
            offset_x = obj['offsetX']
            offset_y = obj['offsetY']
            speed_x = obj['speed_x']
            speed_y = obj['speed_y']
            points_1, points_2, points_3, points_4 = obj['points_edgebox']
            size_width = obj['size']['width']
            size_length = obj['size']['length']
            
            one_line = [site_id, date_time, msg_count, timestamp, ref_pos, obj_id, obj_type, 
                       heading, offset_x, offset_y, speed_x, speed_y, points_1, points_2, 
                       points_3, points_4, size_width, size_length]
            df_list.append(one_line)
    
    client.close()
    
    # DataFrame 생성
    df = pd.DataFrame(df_list, columns=['site_id', 'date_time', 'msg_count', 'timestamp', 'ref_pos', 
                                       'obj_id', 'obj_type', 'heading', 'offset_x', 'offset_y', 
                                       'speed_x', 'speed_y', 'points_1', 'points_2', 'points_3', 
                                       'points_4', 'size_width', 'size_length'])
    
    return df

def flatten_json_to_csv(json_paths):
    """
    차량 데이터가 담긴 JSON 파일들을 평탄화하여 DataFrame으로 변환
    
    Parameters:
    - json_paths: JSON 파일 경로 리스트
    
    Returns:
    - DataFrame: 평탄화된 데이터프레임
    """
    df_list = []
    
    for json_path in tqdm(json_paths, desc="Processing JSON files"):
        # JSON 파일 로드
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # JSON 구조 순회하며 데이터 평탄화
        for msg in data:
            site_id = msg['Site_ID']
            date_time = msg['datetime']['$date']
            msg_count = msg['msgCnt']
            timestamp = msg['sDSMTimeStamp']
            ref_pos = msg['refPos']
            
            for obj in msg['DetectedObjectList']:
                obj_id = obj['objectID']
                obj_type = obj['objType']
                if obj_type != 1:  # 차량만 선택
                    continue
                    
                heading = obj['heading']
                offset_x = obj['offsetX']
                offset_y = obj['offsetY']
                speed_x = obj['speed_x']
                speed_y = obj['speed_y']
                points_1, points_2, points_3, points_4 = obj['points_edgebox']
                size_width = obj['size']['width']
                size_length = obj['size']['length']
                
                one_line = [site_id, date_time, msg_count, timestamp, ref_pos, obj_id, obj_type, 
                           heading, offset_x, offset_y, speed_x, speed_y, points_1, points_2, 
                           points_3, points_4, size_width, size_length]
                df_list.append(one_line)
    
    # DataFrame 생성
    df = pd.DataFrame(df_list, columns=['site_id', 'date_time', 'msg_count', 'timestamp', 'ref_pos', 
                                       'obj_id', 'obj_type', 'heading', 'offset_x', 'offset_y', 
                                       'speed_x', 'speed_y', 'points_1', 'points_2', 'points_3', 
                                       'points_4', 'size_width', 'size_length'])
    
    return df

def preprocess_data(df):
    """
    데이터 전처리: 스케일 변환, 회전 변환, 속도 계산 등
    
    Parameters:
    - df: 원본 데이터프레임
    
    Returns:
    - DataFrame: 전처리된 데이터프레임
    """
    # cm -> m 변환
    df['offset_x'] = df['offset_x'] / 100
    df['offset_y'] = df['offset_y'] / 100
    
    # 80도 회전 변환 (반시계방향)
    theta = np.radians(80)
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], 
                               [np.sin(theta), np.cos(theta)]])

    # 회전 변환 적용
    points = np.column_stack((df['offset_x'], df['offset_y']))
    rotated_points = np.dot(points, rotation_matrix.T)

    # 회전된 좌표를 새로운 컬럼으로 추가
    df['rotated_x'] = rotated_points[:, 0]
    df['rotated_y'] = rotated_points[:, 1]

    # 속도 계산 (m/s)
    speed = np.sqrt(df['speed_x']**2 + df['speed_y']**2)
    df['speed'] = speed

    # heading 변환 (deg -> rad)
    df['heading_deg'] = df['heading'] / 100
    df['heading_rad'] = df['heading_deg'] * np.pi / 180

    # 시간대 변환
    df['datetime_kst'] = pd.to_datetime(df['timestamp'].apply(lambda x: int(x)), unit='ms').dt.tz_localize('UTC').dt.tz_convert('Asia/Seoul')

    # 불필요한 컬럼 제거
    df.drop(columns=['offset_x', 'offset_y', 'heading'], inplace=True)
    
    return df

def filter_data(df):
    """
    데이터 필터링: 크기 기반 필터링, 짧은 객체 제거
    
    Parameters:
    - df: 전처리된 데이터프레임
    
    Returns:
    - DataFrame: 필터링된 데이터프레임
    """
    # 객체별 width, length 최댓값 계산
    obj_size_medians = df.groupby('obj_id').agg({
        'size_width': 'max',
        'size_length': 'max'
    }).reset_index()

    # 크기 기반 필터링 (정상 크기 객체만 선택)
    df_large = obj_size_medians[(obj_size_medians['size_width'] <= 200) & 
                                 (obj_size_medians['size_length'] <= 520)]

    # 필터링된 객체만 선택
    df = df[df['obj_id'].isin(df_large['obj_id'])]
    
    # 너무 짧은 객체 제거 (40프레임 이하)
    counts = df.groupby('obj_id').size()
    obj_ids_40 = counts[counts <= 40].index
    df = df[~df['obj_id'].isin(obj_ids_40)]
    
    return df

def create_object_csv(df):
    """
    객체 정보 CSV 생성
    
    Parameters:
    - df: 필터링된 데이터프레임
    
    Returns:
    - DataFrame: 객체 정보 데이터프레임
    """
    row_object = []
    
    for obj_id in df['obj_id'].unique():
        df_object = df[df['obj_id'] == obj_id]
        
        # 객체 정보 추출
        obj_id = df_object['obj_id'].iloc[0]
        source_id = 9750122  # ASCII
        equip_type = 1
        object_type = 1
        veh_width = df_object['size_width'].median()
        veh_length = df_object['size_length'].median()
        start_date = df_object['datetime_kst'].iloc[0]
        
        row_object.append([obj_id, source_id, equip_type, object_type, veh_width, veh_length, start_date])
    
    object_df = pd.DataFrame(row_object, columns=['ObjectID', 'SourceID', 'EquipmentType', 'ObjectType', 'VehicleWidth', 'VehicleLength', 'StartDate'])
    
    return object_df

def create_track_csv(df):
    """
    트래킹 정보 CSV 생성
    
    Parameters:
    - df: 필터링된 데이터프레임
    
    Returns:
    - DataFrame: 트래킹 정보 데이터프레임
    """
    # date_time 기준으로 정렬
    df = df.sort_values('date_time')

    # 중복되지 않는 유니크한 date_time 값 추출
    unique_times = df['date_time'].unique()
    unique_times = np.sort(unique_times)

    # date_time 값과 FrameCount 매핑 생성
    frame_count_map = {time: idx for idx, time in enumerate(unique_times)}

    # FrameCount 컬럼 생성
    df['FrameCount'] = df['date_time'].map(frame_count_map)
    
    track_df = pd.DataFrame(columns=['FrameCount', 'ObjectID', 'VehicleClass', 'DistanceX', 'DistanceY', 
                                     'Speed', 'Heading', 'AccelerationX', 'AccelerationY'])

    for obj_id in tqdm(df['obj_id'].unique(), desc="Creating track data"):
        df_object = df[df['obj_id'] == obj_id]
        df_object = df_object.sort_values(by='datetime_kst')
        df_object = df_object.reset_index(drop=True)
        
        # 필요한 컬럼 매핑 및 이름 변경
        obj_track_df = df_object[['FrameCount', 'obj_id', 'rotated_x', 'rotated_y', 'heading_deg']].copy()
        obj_track_df = obj_track_df.rename(columns={
            'obj_id': 'ObjectID',
            'rotated_x': 'DistanceX', 
            'rotated_y': 'DistanceY',
            'heading_deg': 'Heading'
        })
        
        # VehicleClass 추가 (모든 객체를 차량으로 가정)
        obj_track_df['VehicleClass'] = 100
        
        # 속도 계산
        speed_x = df_object['speed_x']
        speed_y = df_object['speed_y']
        obj_track_df['Speed'] = np.sqrt(speed_x**2 + speed_y**2)
        
        # 가속도 계산
        diff_speed_x = speed_x.diff()
        diff_speed_y = speed_y.diff()
        dt = df_object['datetime_kst'].diff().dt.total_seconds()
        
        obj_track_df['AccelerationX'] = 0.0  # 첫 프레임은 0으로 초기화
        obj_track_df.loc[1:, 'AccelerationX'] = diff_speed_x[1:] / dt[1:]   
        obj_track_df['AccelerationY'] = 0.0  # 첫 프레임은 0으로 초기화
        obj_track_df.loc[1:, 'AccelerationY'] = diff_speed_y[1:] / dt[1:]
        
        # 컬럼 순서 맞추기
        obj_track_df = obj_track_df[['FrameCount', 'ObjectID', 'VehicleClass', 'DistanceX', 
                                    'DistanceY', 'Speed', 'Heading', 'AccelerationX', 'AccelerationY']]
        
        # track_df에 추가
        track_df = pd.concat([track_df, obj_track_df], ignore_index=True)
    
    return track_df

def process_vehicle_data_from_json(json_dir, output_dir, data_name):
    """
    JSON 파일에서 차량 데이터 전처리
    
    Parameters:
    - json_dir: JSON 파일들이 있는 디렉토리 경로
    - output_dir: 결과물을 저장할 디렉토리 경로
    - data_name: 데이터셋 이름
    
    Returns:
    - tuple: (전체 object 개수, 총 tracking row 개수)
    """
    # 1. 출력 디렉토리 설정
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 2. JSON 파일 경로들 수집
    json_paths = glob.glob(os.path.join(json_dir, "*.json"))
    if not json_paths:
        raise ValueError(f"No JSON files found in {json_dir}")
    
    print(f"Found {len(json_paths)} JSON files")
    
    # 3. JSON 데이터 평탄화
    print("Flattening JSON data...")
    df = flatten_json_to_csv(json_paths)
    print(f"Initial data shape: {df.shape}")
    
    return process_vehicle_data_common(df, output_dir, data_name)

def process_vehicle_data_from_mongodb(connection_string, output_dir, data_name, start_date=None, end_date=None, site_id=None):
    """
    MongoDB에서 차량 데이터 전처리
    
    Parameters:
    - connection_string: MongoDB 연결 문자열
    - output_dir: 결과물을 저장할 디렉토리 경로
    - data_name: 데이터셋 이름
    - start_date: 시작 날짜 (문자열, YYYY-MM-DD 형식)
    - end_date: 종료 날짜 (문자열, YYYY-MM-DD 형식)
    - site_id: 특정 사이트 ID (선택사항)
    
    Returns:
    - tuple: (전체 object 개수, 총 tracking row 개수)
    """
    # 1. 출력 디렉토리 설정
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 2. 날짜 변환 (timestamp in milliseconds)
    start_dt = None
    end_dt = None
    if start_date:
        start_dt = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp() * 1000)
    if end_date:
        end_dt = int(datetime.strptime(end_date, "%Y-%m-%d").timestamp() * 1000)
    
    # 3. MongoDB에서 데이터 불러오기
    print("Loading data from MongoDB...")
    df = load_from_mongodb(connection_string, start_dt, end_dt, site_id)
    print(f"Initial data shape: {df.shape}")
    
    return process_vehicle_data_common(df, output_dir, data_name)

def process_vehicle_data_common(df, output_dir, data_name):
    """
    공통 데이터 처리 함수
    
    Parameters:
    - df: 원본 데이터프레임
    - output_dir: 결과물을 저장할 디렉토리 경로
    - data_name: 데이터셋 이름
    
    Returns:
    - tuple: (전체 object 개수, 총 tracking row 개수)
    """
    
    # 4. 데이터 전처리
    print("Preprocessing data...")
    df = preprocess_data(df)
    
    # 5. 데이터 필터링
    print("Filtering data...")
    df = filter_data(df)
    print(f"Filtered data shape: {df.shape}")
    
    # 6. 객체 정보 CSV 생성
    print("Creating object CSV...")
    object_df = create_object_csv(df)
    
    # 7. 트래킹 정보 CSV 생성
    print("Creating track CSV...")
    track_df = create_track_csv(df)
    
    # 8. 숫자형 컬럼 반올림
    numeric_columns = track_df.select_dtypes(include=['float64', 'float32']).columns
    track_df[numeric_columns] = track_df[numeric_columns].round(3)
    
    numeric_columns = object_df.select_dtypes(include=['float64', 'float32']).columns  
    object_df[numeric_columns] = object_df[numeric_columns].round(3)
    
    # 9. CSV 파일 저장
    object_path = output_dir / f"{data_name}_object.csv"
    track_path = output_dir / f"{data_name}_track.csv"
    
    object_df.to_csv(object_path, index=False)
    track_df.to_csv(track_path, index=False)
    
    print(f"Object CSV saved to: {object_path}")
    print(f"Track CSV saved to: {track_path}")
    
    # 10. 결과 통계 반환
    total_objects = object_df['ObjectID'].nunique()
    total_tracking_rows = len(track_df)
    
    return total_objects, total_tracking_rows

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Vehicle data preprocessing script")
    
    # 데이터 소스 선택
    parser.add_argument("--source", type=str, required=True, 
                       choices=["json", "mongodb"], 
                       help="Data source: 'json' or 'mongodb'")
    
    # 공통 인자
    parser.add_argument("--output_dir", type=str, required=True, 
                       help="Output directory for CSV files")
    parser.add_argument("--data_name", type=str, required=True, 
                       help="Dataset name for output files")
    
    # JSON 관련 인자
    parser.add_argument("--json_dir", type=str, 
                       help="Directory containing JSON files (required when source=json)")
    
    # MongoDB 관련 인자
    parser.add_argument("--start_date", type=str, 
                       help="Start date (YYYY-MM-DD format, optional)")
    parser.add_argument("--end_date", type=str, 
                       help="End date (YYYY-MM-DD format, optional)")
    parser.add_argument("--site_id", type=str, 
                       help="Site ID filter (optional)")
    
    args = parser.parse_args()

    if args.source == "json":
        if not args.json_dir:
            parser.error("--json_dir is required when source=json")
        
        total_objects, total_tracking_rows = process_vehicle_data_from_json(
            args.json_dir, 
            args.output_dir, 
            args.data_name
        )
        
    elif args.source == "mongodb":
        
        total_objects, total_tracking_rows = process_vehicle_data_from_mongodb(
            MONGO_URI,
            args.output_dir,
            args.data_name,
            args.start_date,
            args.end_date,
            args.site_id
        )
    
    print(f"\n=== Processing Complete ===")
    print(f"추출된 주행 차량 수: {total_objects}")
    print(f"주행 메시지 수: {total_tracking_rows}")
