import os
import shutil
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--root_dir",
        type=str,
        default=None,
        required=True,
        help="Path to source directory.",
    )

    parser.add_argument(
        "--save_dir",
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

if __name__=="__main__":
    args = parse_args()

    for uc in args.use_case:
        use_case_folder=os.path.join(args.root_dir, uc)
        for split in os.listdir(use_case_folder):
            if args.mode and split=='test':
                continue
            elif not args.mode and split in ['train', 'val']:
                continue

            src_raw_folder_path = os.path.join(use_case_folder, split, 'raw')
            if not os.path.exists(src_raw_folder_path):
                continue
            
            for raw_name in os.listdir(src_raw_folder_path):
                src_raw_file_path = os.path.join(src_raw_folder_path, raw_name)
                for name in os.listdir(src_raw_file_path):
                    save_raw_folder_path = os.path.join(args.save_dir, split, 'raw', '{}_{}'.format(uc, raw_name))
                    if not os.path.exists(save_raw_folder_path):
                        os.makedirs(save_raw_folder_path, exist_ok=True)
                    shutil.copy(os.path.join(src_raw_file_path,name), os.path.join(save_raw_folder_path, name.replace(raw_name,'{}_{}'.format(uc, raw_name))))