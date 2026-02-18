# train/val
python 00_merge_sumo.py --root_dir /workspace/Dataset/IG --use_case "SS_ped" --mode
python 01_agent_SUMOtoAV2.py --src_dir /workspace/Dataset/IG --dst_dir /workspace/Desktop/data_sumo --use_case "SS_ped" --mode --sample_step 10 --input_frame 10
python 02_json_copy.py --src_dir /workspace/Dataset/IG --dst_dir /workspace/Desktop/data_sumo --use_case "SS_ped" --mode