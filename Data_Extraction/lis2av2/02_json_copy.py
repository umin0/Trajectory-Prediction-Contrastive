import os
import shutil
import argparse
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--src_dir",
        type=str,
        default=None,
        required=True,
        help="Path to source directory.",
    )

    parser.add_argument(
        "--dst_dir",
        type=str,
        default=None,
        required=True,
        help="Path to destination directory.",
    )
    parser.add_argument(
        "--use_case",
        type=str,
        nargs='+',
        default=["SS"],
        required=False,
        help="Use Case for train / validation dataset",
    )
    parser.add_argument(
        "--mode",
        action='store_true',
        required=False,
        help="If mode is true, run the train/validation process else run the test process.",
    )
    args = parser.parse_args()
    return args

def copy_json_to_subfolders_with_new_name(source_file_path, dest_folder):
    # 대상 폴더의 모든 하위 폴더 목록을 가져오기
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)
    subfolders = [os.path.join(dest_folder, name) for name in os.listdir(dest_folder) if os.path.isdir(os.path.join(dest_folder, name))]

    # 원본 JSON 파일을 각 하위 폴더로 복사하며 파일명 변경
    for folder in tqdm(subfolders, desc=f"Copying JSON files {source_file_path}"):
        folder_name = os.path.basename(folder)
        new_file_name = f"log_map_archive_{folder_name}.json"
        dest_file_path = os.path.join(folder, new_file_name)
        shutil.copyfile(source_file_path, dest_file_path)

if __name__=="__main__":
    args = parse_args()

    for uc in args.use_case:
        source_file_path = os.path.join(args.src_dir, '{}/{}_map.json'.format(uc,uc))

        if args.mode:
            train_dest_folder = os.path.join(args.dst_dir,'{}/train/raw'.format(uc))
            copy_json_to_subfolders_with_new_name(source_file_path, train_dest_folder)

            val_dest_folder = os.path.join(args.dst_dir,'{}/val/raw'.format(uc))
            copy_json_to_subfolders_with_new_name(source_file_path, val_dest_folder)
        else:
            test_dest_folder = os.path.join(args.dst_dir,'{}/test/raw'.format(uc))
            copy_json_to_subfolders_with_new_name(source_file_path, test_dest_folder)