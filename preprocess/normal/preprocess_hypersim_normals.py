# Copyright 2023-2025 Marigold Team, ETH Zürich. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# --------------------------------------------------------------------------
# More information about Marigold:
#   https://marigoldmonodepth.github.io
#   https://marigoldcomputervision.github.io
# Efficient inference pipelines are now part of diffusers:
#   https://huggingface.co/docs/diffusers/using-diffusers/marigold_usage
#   https://huggingface.co/docs/diffusers/api/pipelines/marigold
# Examples of trained models and live demos:
#   https://huggingface.co/prs-eth
# Related projects:
#   https://rollingdepth.github.io/
#   https://marigolddepthcompletion.github.io/
# Citation (BibTeX):
#   https://github.com/prs-eth/Marigold#-citation
# If you find Marigold useful, we kindly ask you to cite our papers.
# --------------------------------------------------------------------------

import argparse
import cv2
import h5py
import numpy as np
import os
import pandas as pd
import sklearn.preprocessing
from tqdm import tqdm
import multiprocessing as mp
from functools import partial

from pylab import count_nonzero, clip, np


# Adapted from https://github.com/apple/ml-hypersim/blob/main/code/python/tools/scene_generate_images_tonemap.py
def tone_map(rgb, entity_id_map):
    assert (entity_id_map != 0).all()

    gamma = 1.0 / 2.2  # standard gamma correction exponent
    inv_gamma = 1.0 / gamma
    percentile = (
        90  # we want this percentile brightness value in the unmodified image...
    )
    brightness_nth_percentile_desired = 0.8  # ...to be this bright after scaling

    valid_mask = entity_id_map != -1

    if count_nonzero(valid_mask) == 0:
        scale = 1.0  # if there are no valid pixels, then set scale to 1.0
    else:
        brightness = (
            0.3 * rgb[:, :, 0] + 0.59 * rgb[:, :, 1] + 0.11 * rgb[:, :, 2]
        )  # "CCIR601 YIQ" method for computing brightness
        brightness_valid = brightness[valid_mask]

        eps = 0.0001  # if the kth percentile brightness value in the unmodified image is less than this, set the scale to 0.0 to avoid divide-by-zero
        brightness_nth_percentile_current = np.percentile(brightness_valid, percentile)

        if brightness_nth_percentile_current < eps:
            scale = 0.0
        else:
            # Snavely uses the following expression in the code at https://github.com/snavely/pbrs_tonemapper/blob/master/tonemap_rgbe.py:
            # scale = np.exp(np.log(brightness_nth_percentile_desired)*inv_gamma - np.log(brightness_nth_percentile_current))
            #
            # Our expression below is equivalent, but is more intuitive, because it follows more directly from the expression:
            # (scale*brightness_nth_percentile_current)^gamma = brightness_nth_percentile_desired

            scale = (
                np.power(brightness_nth_percentile_desired, inv_gamma)
                / brightness_nth_percentile_current
            )

    rgb_color_tm = np.power(np.maximum(scale * rgb, 0), gamma)
    rgb_color_tm = clip(rgb_color_tm, 0, 1)
    return rgb_color_tm



IMG_WIDTH = 1024
IMG_HEIGHT = 768
FOCAL_LENGTH = 886.81


def process_single_row(row_data, dataset_dir, split_output_dir):
    """Process a single row of data"""
    try:
        i, row = row_data
        
        # Load data
        rgb_path = os.path.join(
            row.scene_name,
            "images",
            f"scene_{row.camera_name}_final_hdf5",
            f"frame.{row.frame_id:04d}.color.hdf5",
        )
        normal_cam_path = os.path.join(
            row.scene_name,
            "images",
            f"scene_{row.camera_name}_geometry_hdf5",
            f"frame.{row.frame_id:04d}.normal_cam.hdf5",
        )
        normal_world_path = os.path.join(
            row.scene_name,
            "images",
            f"scene_{row.camera_name}_geometry_hdf5",
            f"frame.{row.frame_id:04d}.normal_world.hdf5",
        )
        position_path = os.path.join(
            row.scene_name,
            "images",
            f"scene_{row.camera_name}_geometry_hdf5",
            f"frame.{row.frame_id:04d}.position.hdf5",
        )
        camera_keyframe_positions_path = os.path.join(
            row.scene_name,
            "_detail",
            f"{row.camera_name}",
            "camera_keyframe_positions.hdf5",
        )
        render_entity_id_path = os.path.join(
            row.scene_name,
            "images",
            f"scene_{row.camera_name}_geometry_hdf5",
            f"frame.{row.frame_id:04d}.render_entity_id.hdf5",
        )
        
        assert os.path.exists(os.path.join(dataset_dir, rgb_path))
        assert os.path.exists(os.path.join(dataset_dir, normal_cam_path))
        assert os.path.exists(os.path.join(dataset_dir, normal_world_path))

        with h5py.File(os.path.join(dataset_dir, rgb_path), "r") as f:
            rgb = np.array(f["dataset"]).astype(float)
        with h5py.File(os.path.join(dataset_dir, render_entity_id_path), "r") as f:
            render_entity_id = np.array(f["dataset"]).astype(int)
        with h5py.File(os.path.join(dataset_dir, normal_cam_path), "r") as f:
            normal_cam = np.array(f["dataset"]).astype(float)  # [H,W,3]
        with h5py.File(os.path.join(dataset_dir, position_path), "r") as f:
            position = np.array(f["dataset"]).astype(float)
        with h5py.File(os.path.join(dataset_dir, normal_world_path), "r") as f:
            normal_world = np.array(f["dataset"]).astype(float)
        with h5py.File(
            os.path.join(dataset_dir, camera_keyframe_positions_path), "r"
        ) as f:
            camera_keyframe_positions = np.array(f["dataset"]).astype(float)

        camera_position = camera_keyframe_positions[int(row.frame_id)]

        # Tone map
        rgb_color_tm = tone_map(rgb, render_entity_id)
        rgb_int = (rgb_color_tm * 255).astype(np.uint8)  # [H, W, RGB]

        # Pre-process normals
        # 1) normalize to unit length
        # 2) invert the wrong normals that are pointing the same way as the camera, instead of against it
        if np.any(
            np.isnan(normal_cam)
        ):  # skip if the normal map contains Nan values
            print(f"Skipping row {i}: normal map contains NaN values")
            return None
        else:
            # make sure normals are correctly normalized
            normal_cam_1d_ = normal_cam.reshape(-1, 3)
            normal_cam_1d_ = sklearn.preprocessing.normalize(normal_cam_1d_)
            normal_cam = normal_cam_1d_.reshape(normal_cam.shape)

            # scene ai_051_004 has a few wrong -inf values for camera position
            # replace them with a neighboring value from same channel
            if np.any(np.isinf(position)):
                inf_indices = np.where(np.isinf(position))
                for idx in zip(*inf_indices):
                    h, w, ch = idx
                    if h == 0:
                        position[h, w, ch] = position[h + 1, w, ch]
                    else:
                        position[h, w, ch] = position[h - 1, w, ch]

            position_1d_ = position.reshape(-1, 3)
            normal_world_1d_ = normal_world.reshape(-1, 3)

            # check if normals are pointing the same way as the camera, instead of against it
            surface_to_cam_world_normalized_1d_ = sklearn.preprocessing.normalize(
                camera_position - position_1d_
            )
            n_dot_v_1d_ = np.sum(
                normal_world_1d_ * surface_to_cam_world_normalized_1d_, axis=1
            )
            normal_back_facing_mask_1d_ = n_dot_v_1d_ < -(1e-3)
            normal_back_facing_mask = normal_back_facing_mask_1d_.reshape(
                normal_world.shape[0], normal_world.shape[1]
            )
            # invert wrong-facing normals
            normal_cam[normal_back_facing_mask] = (
                -1 * normal_cam[normal_back_facing_mask]
            )

        scene_path = row.scene_name
        scene_dir = os.path.join(split_output_dir, row.scene_name)
        if not os.path.exists(scene_dir):
            os.makedirs(scene_dir, exist_ok=True)

        # Save RGB
        rgb_name = f"rgb_{row.camera_name}_fr{row.frame_id:04d}.png"
        rgb_save_path = os.path.join(scene_path, rgb_name)
        cv2.imwrite(
            os.path.join(split_output_dir, rgb_save_path),
            cv2.cvtColor(rgb_int, cv2.COLOR_RGB2BGR),
        )

        # save normals
        normal_cam_name = f"normal_cam_{row.camera_name}_fr{row.frame_id:04d}.npy"
        normal_cam_save_path = os.path.join(scene_path, normal_cam_name)
        np.save(os.path.join(split_output_dir, normal_cam_save_path), normal_cam)

        # Return results for meta data update
        return {
            'index': i,
            'rgb_path': rgb_save_path,
            'rgb_mean': np.mean(rgb_int),
            'rgb_std': np.std(rgb_int),
            'rgb_min': np.min(rgb_int),
            'rgb_max': np.max(rgb_int),
            'normal_path': normal_cam_save_path
        }
        
    except Exception as e:
        print(f"Error processing row {i}: {str(e)}")
        return None


if "__main__" == __name__:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split_csv",
        type=str,
        default="preprocess/depth/metadata_images_split_scene_v1.csv",
    )
    parser.add_argument("--dataset_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--num_processes", type=int, default=mp.cpu_count(), 
                       help="Number of processes to use (default: number of CPU cores)")

    args = parser.parse_args()

    split_csv = args.split_csv
    dataset_dir = args.dataset_dir
    output_dir = args.output_dir
    num_processes = args.num_processes

    # %%
    raw_meta_df = pd.read_csv(split_csv)
    meta_df = raw_meta_df[raw_meta_df.included_in_public_release].copy()

    # %%
    for split in ["train", "val", "test"]:
        split_output_dir = os.path.join(output_dir, split)
        os.makedirs(split_output_dir, exist_ok=True)

        split_meta_df = meta_df[meta_df.split_partition_name == split].copy()
        split_meta_df["rgb_path"] = None
        split_meta_df["rgb_mean"] = np.nan
        split_meta_df["rgb_std"] = np.nan
        split_meta_df["rgb_min"] = np.nan
        split_meta_df["rgb_max"] = np.nan
        split_meta_df["normal_path"] = None

        # Prepare data for multiprocessing
        row_data_list = list(split_meta_df.iterrows())
        
        # Create partial function with fixed arguments
        process_func = partial(process_single_row, 
                              dataset_dir=dataset_dir, 
                              split_output_dir=split_output_dir)

        # Process with multiprocessing
        print(f"Processing {len(row_data_list)} samples for {split} split with {num_processes} processes...")
        with mp.Pool(processes=num_processes) as pool:
            results = list(tqdm(
                pool.imap(process_func, row_data_list),
                total=len(row_data_list),
                desc=f"Processing {split}"
            ))

        # Update metadata with results
        for result in results:
            if result is not None:
                i = result['index']
                split_meta_df.at[i, "rgb_path"] = result['rgb_path']
                split_meta_df.at[i, "rgb_mean"] = result['rgb_mean']
                split_meta_df.at[i, "rgb_std"] = result['rgb_std']
                split_meta_df.at[i, "rgb_min"] = result['rgb_min']
                split_meta_df.at[i, "rgb_max"] = result['rgb_max']
                split_meta_df.at[i, "normal_path"] = result['normal_path']

        # Filter out failed samples for file writing
        successful_rows = split_meta_df.dropna(subset=['rgb_path', 'normal_path'])

        with open(
            os.path.join(split_output_dir, f"hypersim_filtered_{split}.txt"), "w+"
        ) as f:
            lines = successful_rows.apply(
                lambda r: f"{r['rgb_path']} {r['normal_path']}", axis=1
            ).tolist()
            f.writelines("\n".join(lines))

        with open(
            os.path.join(split_output_dir, f"filename_meta_{split}.csv"), "w+"
        ) as f:
            split_meta_df.to_csv(f, header=True)

        print(f"Processed {len(successful_rows)} successful samples out of {len(split_meta_df)} total samples for {split} split")

    print("Preprocess finished")
