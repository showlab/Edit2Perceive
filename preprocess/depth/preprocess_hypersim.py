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


# According to https://github.com/apple/ml-hypersim/issues/9
def dist_2_depth(width, height, flt_focal, distance):
    img_plane_x = (
        np.linspace((-0.5 * width) + 0.5, (0.5 * width) - 0.5, width)
        .reshape(1, width)
        .repeat(height, 0)
        .astype(np.float32)[:, :, None]
    )
    img_plane_y = (
        np.linspace((-0.5 * height) + 0.5, (0.5 * height) - 0.5, height)
        .reshape(height, 1)
        .repeat(width, 1)
        .astype(np.float32)[:, :, None]
    )
    img_plane_z = np.full([height, width, 1], flt_focal, np.float32)
    img_plane = np.concatenate([img_plane_x, img_plane_y, img_plane_z], 2)

    depth = distance / np.linalg.norm(img_plane, 2, 2) * flt_focal
    return depth

import argparse
import cv2
import h5py
import numpy as np
import os
import pandas as pd
from tqdm import tqdm
import multiprocessing as mp

# from hypersim_util import dist_2_depth, tone_map

IMG_WIDTH = 1024
IMG_HEIGHT = 768
FOCAL_LENGTH = 886.81

def process_item(args):
    """
    Worker function to process one row/frame.
    Returns a dict with index and all values to write back to the dataframe.
    """
    (
        idx,
        scene_name,
        camera_name,
        frame_id,
        dataset_dir,
        split_output_dir,
        IMG_WIDTH,
        IMG_HEIGHT,
        FOCAL_LENGTH,
    ) = args

    # Build input file paths (relative to dataset_dir)
    dataset_rgb_path = os.path.join(
        scene_name,
        "images",
        f"scene_{camera_name}_final_hdf5",
        f"frame.{frame_id:04d}.color.hdf5",
    )
    dist_path = os.path.join(
        scene_name,
        "images",
        f"scene_{camera_name}_geometry_hdf5",
        f"frame.{frame_id:04d}.depth_meters.hdf5",
    )
    render_entity_id_path = os.path.join(
        scene_name,
        "images",
        f"scene_{camera_name}_geometry_hdf5",
        f"frame.{frame_id:04d}.render_entity_id.hdf5",
    )

    # sanity checks (will raise AssertionError if missing)
    assert os.path.exists(os.path.join(dataset_dir, dataset_rgb_path))
    assert os.path.exists(os.path.join(dataset_dir, dist_path))

    # Read files
    with h5py.File(os.path.join(dataset_dir, dataset_rgb_path), "r") as f:
        rgb = np.array(f["dataset"]).astype(float)
    with h5py.File(os.path.join(dataset_dir, dist_path), "r") as f:
        dist_from_center = np.array(f["dataset"]).astype(float)
    with h5py.File(os.path.join(dataset_dir, render_entity_id_path), "r") as f:
        render_entity_id = np.array(f["dataset"]).astype(int)

    # Tone map
    rgb_color_tm = tone_map(rgb, render_entity_id)
    rgb_int = (rgb_color_tm * 255).astype(np.uint8)  # [H, W, RGB]

    # Distance -> depth
    plane_depth = dist_2_depth(IMG_WIDTH, IMG_HEIGHT, FOCAL_LENGTH, dist_from_center)
    valid_mask = render_entity_id != -1

    # Record invalid ratio
    invalid_ratio = (np.prod(valid_mask.shape) - valid_mask.sum()) / np.prod(
        valid_mask.shape
    )
    plane_depth[~valid_mask] = 0

    # Ensure scene directory exists under split_output_dir (avoid race with exist_ok=True)
    scene_out_dir = os.path.join(split_output_dir, scene_name)
    os.makedirs(scene_out_dir, exist_ok=True)

    # Save RGB png
    rgb_name = f"rgb_{camera_name}_fr{frame_id:04d}.png"
    out_rgb_relpath = os.path.join(scene_name, rgb_name)
    out_rgb_full = os.path.join(split_output_dir, out_rgb_relpath)
    cv2.imwrite(out_rgb_full, cv2.cvtColor(rgb_int, cv2.COLOR_RGB2BGR))

    # Save depth png (scale to mm and uint16)
    plane_depth_mm = (plane_depth * 1000.0).astype(np.uint16)
    depth_name = f"depth_plane_{camera_name}_fr{frame_id:04d}.png"
    out_depth_relpath = os.path.join(scene_name, depth_name)
    out_depth_full = os.path.join(split_output_dir, out_depth_relpath)
    cv2.imwrite(out_depth_full, plane_depth_mm)

    # Compute statistics (depth restored to meters)
    restored_depth = plane_depth_mm.astype(np.float32) / 1000.0

    result = {
        "index": idx,
        "rgb_path": out_rgb_relpath,
        "rgb_mean": float(np.mean(rgb_int)),
        "rgb_std": float(np.std(rgb_int)),
        "rgb_min": int(np.min(rgb_int)),
        "rgb_max": int(np.max(rgb_int)),
        "depth_path": out_depth_relpath,
        "depth_mean": float(np.mean(restored_depth)),
        "depth_std": float(np.std(restored_depth)),
        "depth_min": float(np.min(restored_depth)),
        "depth_max": float(np.max(restored_depth)),
        "invalid_ratio": float(invalid_ratio),
    }
    return result

if "__main__" == __name__:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split_csv",
        type=str,
        default="preprocess/depth/metadata_images_split_scene_v1.csv",
    )
    parser.add_argument("--dataset_dir", type=str, default="data/Hypersim/raw_data")
    parser.add_argument("--output_dir", type=str, default="data/Hypersim/processed")

    args = parser.parse_args()

    split_csv = args.split_csv
    dataset_dir = args.dataset_dir
    output_dir = args.output_dir

    # %%
    raw_meta_df = pd.read_csv(split_csv)
    meta_df = raw_meta_df[raw_meta_df.included_in_public_release].copy()

    # %%
    # create top-level output dir if not present (preserve intention of original script)
    os.makedirs(output_dir, exist_ok=True)

    for split in ["train", "val", "test"]:
        split_output_dir = os.path.join(output_dir, split)
        # original code used os.makedirs(split_output_dir) which would error if exists;
        # using exist_ok=True is safe and avoids failures on re-run.
        os.makedirs(split_output_dir, exist_ok=True)

        split_meta_df = meta_df[meta_df.split_partition_name == split].copy()
        split_meta_df["rgb_path"] = None
        split_meta_df["rgb_mean"] = np.nan
        split_meta_df["rgb_std"] = np.nan
        split_meta_df["rgb_min"] = np.nan
        split_meta_df["rgb_max"] = np.nan
        split_meta_df["depth_path"] = None
        split_meta_df["depth_mean"] = np.nan
        split_meta_df["depth_std"] = np.nan
        split_meta_df["depth_min"] = np.nan
        split_meta_df["depth_max"] = np.nan
        split_meta_df["invalid_ratio"] = np.nan

        # Prepare tasks: use the dataframe's index to allow writing back exactly where original wrote.
        tasks = []
        for i, row in split_meta_df.iterrows():
            tasks.append(
                (
                    i,
                    row.scene_name,
                    row.camera_name,
                    int(row.frame_id),
                    dataset_dir,
                    split_output_dir,
                    IMG_WIDTH,
                    IMG_HEIGHT,
                    FOCAL_LENGTH,
                )
            )

        # Use multiprocessing with spawn context (safer for libraries like h5py)
        ctx = mp.get_context("spawn")
        # you can tune processes=None to specific number like processes=mp.cpu_count()
        with ctx.Pool() as pool:
            # imap_unordered + tqdm to show progress as frames finish
            for res in tqdm(pool.imap_unordered(process_item, tasks), total=len(tasks)):
                idx = res["index"]
                split_meta_df.at[idx, "rgb_path"] = res["rgb_path"]
                split_meta_df.at[idx, "rgb_mean"] = res["rgb_mean"]
                split_meta_df.at[idx, "rgb_std"] = res["rgb_std"]
                split_meta_df.at[idx, "rgb_min"] = res["rgb_min"]
                split_meta_df.at[idx, "rgb_max"] = res["rgb_max"]

                split_meta_df.at[idx, "depth_path"] = res["depth_path"]
                split_meta_df.at[idx, "depth_mean"] = res["depth_mean"]
                split_meta_df.at[idx, "depth_std"] = res["depth_std"]
                split_meta_df.at[idx, "depth_min"] = res["depth_min"]
                split_meta_df.at[idx, "depth_max"] = res["depth_max"]

                split_meta_df.at[idx, "invalid_ratio"] = res["invalid_ratio"]

        # Write filename_list and csv exactly like original
        with open(os.path.join(split_output_dir, f"filename_list_{split}.txt"), "w+") as f:
            lines = split_meta_df.apply(
                lambda r: f"{r['rgb_path']} {r['depth_path']}", axis=1
            ).tolist()
            f.writelines("\n".join(lines))

        with open(os.path.join(split_output_dir, f"filename_meta_{split}.csv"), "w+") as f:
            split_meta_df.to_csv(f, header=True)

    print("Preprocess finished")

