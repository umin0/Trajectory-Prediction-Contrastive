import os
from tqdm import tqdm
import pandas as pd
import random
import math
import argparse
import glob


def parse_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--root_dir",
        type=str,
        default=None,
        required=True,
        help="Path to root preprocessing directory.",
    )
    parser.add_argument(
        "--ratio",
        type=float,
        default=0.9,
        required=False,
        help="The train / validation dataset length ratio.",
    )
    parser.add_argument(
        "--use_case",
        type=str,
        nargs="+",
        default=["A1", "B1", "C1", "D3", "D4"],
        required=False,
        help="Use Case for train / validation dataset",
    )
    parser.add_argument(
        "--mode",
        action="store_true",
        required=False,
        help="If mode is true, run the train/validation process else run the test process.",
    )
    args = parser.parse_args()
    return args


def get_av2_data(combined, track_info, object_info, recording_id):
    combined["recordingId"].append(int(recording_id + 1))

    if (
        not len(combined["trackId"])
        or combined["trackId"][-1] != track_info["ObjectID"]
    ):
        track_lift_time = 0
    else:
        track_lift_time = combined["trackLifetime"][-1] + 1

    combined["trackLifetime"].append(int(track_lift_time))

    combined["trackId"].append(int(track_info["ObjectID"]))
    combined["frame"].append(int(track_info["FrameCount"] - 1))
    combined["xCenter"].append(track_info["DistanceX"])
    combined["yCenter"].append(track_info["DistanceY"])

    heading = math.radians(track_info["Heading"])
    heading = -(heading - math.pi / 2)
    speed = track_info["Speed"]
    combined["xVelocity"].append(speed * math.cos(heading))
    combined["yVelocity"].append(speed * math.sin(heading))
    combined["heading"].append(heading)
    combined["xAcceleration"].append(track_info["AccelerationX"])
    combined["yAcceleration"].append(track_info["AccelerationY"])

    if "LongitudinalManeuver" in track_info.keys():
        combined["LongitudinalManeuver"].append(track_info["LongitudinalManeuver"])
    else:
        combined["LongitudinalManeuver"].append(0)
    if "LateralManeuver" in track_info.keys():
        combined["LateralManeuver"].append(track_info["LateralManeuver"])
    else:
        combined["LateralManeuver"].append(0)

    obj_df = obj_dfs[recording_id]
    obj_info = obj_df[obj_df["ObjectID"] == track_info["ObjectID"]]
    combined["width"] = obj_info["VehicleWidth"].values[0]
    combined["length"] = obj_info["VehicleLength"].values[0]
    return combined


if __name__ == "__main__":
    args = parse_args()

    for site in args.use_case:
        combined = {
            "recordingId": [],
            "trackId": [],
            "frame": [],
            "xCenter": [],
            "yCenter": [],
            "heading": [],
            "xVelocity": [],
            "yVelocity": [],
            "xAcceleration": [],
            "yAcceleration": [],
            "LongitudinalManeuver": [],
            "LateralManeuver": [],
            "trackLifetime": [],
            "width": [],
            "length": [],
        }

        obj_csv_list = glob.glob(os.path.join(args.root_dir, site, "*_object.csv"))
        track_csv_list = glob.glob(os.path.join(args.root_dir, site, "*_track.csv"))
        print(obj_csv_list, track_csv_list)
        if args.mode:
            obj_csv_list = [oc for oc in obj_csv_list if "tc" not in oc]
            track_csv_list = [tc for tc in track_csv_list if "tc" not in tc]
        else:
            obj_csv_list = [oc for oc in obj_csv_list if "tc" in oc]
            track_csv_list = [tc for tc in track_csv_list if "tc" in tc]

        obj_dfs = [pd.read_csv(oc) for oc in obj_csv_list]
        track_dfs = [pd.read_csv(tc) for tc in track_csv_list]

        for idx_df, track_df in enumerate(track_dfs):
            for idx_row, row in tqdm(track_df.iterrows()):
                get_av2_data(combined, row, obj_dfs[idx_df], idx_df)

        combined_df = pd.DataFrame(combined)

        if args.mode:
            unique_recording_ids = set(combined_df["recordingId"])
            train_dfs = []
            val_dfs = []

            for recording_id in unique_recording_ids:
                unique_combined_df = combined_df[
                    combined_df["recordingId"] == recording_id
                ]
                object_ids = set(unique_combined_df["trackId"])
                train_object_ids = random.sample(
                    object_ids, int(len(object_ids) * args.ratio)
                )

                train_df = unique_combined_df[
                    unique_combined_df["trackId"].isin(train_object_ids)
                ]
                val_df = unique_combined_df[
                    ~unique_combined_df["trackId"].isin(train_object_ids)
                ]

                train_dfs.append(train_df)
                val_dfs.append(val_df)

            train_dfs = pd.concat(train_dfs, ignore_index=True)
            val_dfs = pd.concat(val_dfs, ignore_index=True)

            train_path = os.path.join(args.root_dir, site, "TRAIN")
            val_path = os.path.join(args.root_dir, site, "VALI")

            os.makedirs(train_path, exist_ok=True)
            os.makedirs(val_path, exist_ok=True)

            train_dfs.to_csv(
                os.path.join(train_path, "pp_{}__combined.csv".format(site)),
                index=False,
            )
            val_dfs.to_csv(
                os.path.join(val_path, "pp_{}__combined.csv".format(site)), index=False
            )

        else:
            test_path = os.path.join(args.root_dir, site, "TEST")
            os.makedirs(test_path, exist_ok=True)
            combined_df.to_csv(
                os.path.join(test_path, "pp_{}__combined.csv".format(site)), index=False
            )
