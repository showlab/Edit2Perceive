# Copyright (C) 2019 Jin Han Lee
#
# This file is a part of BTS.
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>

from __future__ import absolute_import, division, print_function
from diffsynth.utils.metric import compute_normal_metrics
import os
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
import argparse
import fnmatch
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

def hwc2chw(array):
    return array.transpose(2, 0, 1)
def chw2hwc(array):
    return array.transpose(1, 2, 0)

def convert_arg_line_to_args(arg_line):
    for arg in arg_line.split():
        if not arg.strip():
            continue
        yield arg



def resize_tensor(input_tensor, target_height, target_width):
    """
    使用双线性插值调整深度图像大小
    """
    # 多通道resize
    input_tensor = torch.from_numpy(hwc2chw(input_tensor)).unsqueeze(0).float()
    # 使用双线性插值调整大小
    resized_tensor = F.interpolate(
        input_tensor, 
        size=(target_height, target_width), 
        mode='bilinear', 
        align_corners=True
    )
    # 转换回numpy
    resized = resized_tensor.squeeze().numpy()
    resized = chw2hwc(resized)
    # input_tensor = np.ascontiguousarray(input_tensor)
    # resized = cv2.resize(input_tensor, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
    return resized

def load_image_rgb_or_grayscale(image_path):
    """
    加载图像，支持RGB和灰度图像，统一转换为numpy数组
    """
    if image_path.endswith('.npy'):
        # 如果是npy文件，直接读取
        img_array = np.load(image_path)
    else:
    # 首先尝试用PIL加载，可以更好地处理不同格式
        img = Image.open(image_path)
        img_array = np.array(img)
        
        # 如果是RGBA图像，取前3通道，否则RGB或者Gray则不处理
        if len(img_array.shape) == 3 and img_array.shape[2] == 4:  # RGBA
            img_array = img_array[:, :, :3]  # 去掉Alpha通道
        # 对3通道取均值返回
    return img_array


def test(args):
    global gt_depths, missing_ids, pred_filenames,gt_depths_mask
    gt_depths = []
    gt_depths_mask = []
    missing_ids = set()
    pred_filenames = []
    if getattr(args, 'txt_file_list', None) is not None:
        with open(args.txt_file_list, 'r') as f:
            lines = f.readlines()
        for i,line in enumerate(lines):
            line = line.strip().split()[0]
            if line == '':
                continue
            pred_filenames.append(line.replace(".png",".npy"))
    else:
        for root, dirnames, filenames in os.walk(args.pred_path):
            for pred_filename in fnmatch.filter(filenames, '*.png') + fnmatch.filter(filenames, '*.jpg') + fnmatch.filter(filenames, '*.npy'):
                if 'cmap' in pred_filename or 'gt' in pred_filename:
                    continue
                dirname = root.replace(args.pred_path, '')
                if dirname.startswith('/'):
                    dirname = dirname[1:]
                pred_filenames.append(os.path.join(dirname, pred_filename))

    num_test_samples = len(pred_filenames)
    print(f'Found {num_test_samples} prediction files.')
    pred_depths = []

    for i in tqdm(range(num_test_samples)):
        pred_depth_path = os.path.join(args.pred_path,pred_filenames[i])
        pred_depth = load_image_rgb_or_grayscale(pred_depth_path)

        
        if pred_depth is None:
            print('Missing: %s ' % pred_depth_path)
            missing_ids.add(i)
            continue

        # 预测图像是0-255的relative depth，先转换为float
        pred_depth = pred_depth.astype(np.float32)
        
        pred_depths.append(pred_depth)


    # 加载GT深度图
    if args.dataset == 'nyu' or args.dataset == 'scannet' or args.dataset == 'ibims' or args.dataset == 'oasis':
        for t_id in range(num_test_samples):
            if t_id in missing_ids:
                continue
                
            # 构建GT路径，保持与pred相同的目录结构
            pred_relative_path = pred_filenames[t_id]
            gt_depth_path = os.path.join(args.gt_path, pred_relative_path)
            gt_depth_path = gt_depth_path.replace("_img.npy","_normal.npy")
            depth = load_image_rgb_or_grayscale(gt_depth_path)
            if depth is None:
                print('Missing: %s ' % gt_depth_path)
                missing_ids.add(t_id)
                continue
            gt_depths.append(depth)
    elif args.dataset == 'diode':
        for t_id in range(num_test_samples):
            if t_id in missing_ids:
                continue
                
            # 构建GT路径，保持与pred相同的目录结构
            pred_relative_path = pred_filenames[t_id]
            gt_depth_path = os.path.join(args.gt_path, pred_relative_path)
            gt_depth_path = gt_depth_path.replace(".npy","_normal.npy")
            gt_depth_mask_path = gt_depth_path.replace("_depth.npy","_depth_mask.npy")
            depth = load_image_rgb_or_grayscale(gt_depth_path)
            depth_mask = load_image_rgb_or_grayscale(gt_depth_mask_path)
            if depth is None:
                print('Missing: %s ' % gt_depth_path)
                missing_ids.add(t_id)
                continue
            gt_depths.append(depth)
            gt_depths_mask.append(depth_mask)
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")
    print(f'### Computing errors for {len(gt_depths)} files with {len(missing_ids)} missing' if not gt_depths_mask else 'Computing errors with masks')

    results = eval(pred_depths,args)

    print('Done.')
    return results

def eval(pred_depths,args):
    num_samples = len(pred_depths)
    pred_depths_valid = []
    gt_depths_valid = []

    # 收集有效的预测和GT深度
    gt_idx = 0
    for t_id in range(num_samples):
        if t_id in missing_ids:
            continue

        pred_depths_valid.append(pred_depths[t_id])
        gt_depths_valid.append(gt_depths[gt_idx])
        gt_idx += 1

    num_samples = len(pred_depths_valid)

    mean_angular_error = np.zeros(num_samples, dtype=np.float32)
    median_angular_error = np.zeros(num_samples, dtype=np.float32)
    rmse_angular_error = np.zeros(num_samples, dtype=np.float32)
    sub5_error = np.zeros(num_samples, dtype=np.float32)
    sub7_5_error = np.zeros(num_samples, dtype=np.float32)
    sub11_25_error = np.zeros(num_samples, dtype=np.float32)
    sub22_5_error = np.zeros(num_samples, dtype=np.float32)
    sub30_error = np.zeros(num_samples, dtype=np.float32)
    
    for i in range(num_samples):
        gt_depth = gt_depths_valid[i]
        gt_depth[:,:,0] *= -1
        gt_depth[np.isinf(gt_depth)] = 0
        gt_depth[np.isnan(gt_depth)] = 0
        pred_depth = pred_depths_valid[i]
        pred_depth[np.isinf(pred_depth)] = 0
        pred_depth[np.isnan(pred_depth)] = 0

        # 1. 首先调整预测深度的大小以匹配GT
        if pred_depth.shape != gt_depth.shape:
            pred_depth = resize_tensor(pred_depth, gt_depth.shape[0], gt_depth.shape[1])
        # if i < 5:
        #     H, W, _ = gt_depth.shape
        #     # num_points = 200
        #     # ys = np.random.randint(0, H, size=num_points)
        #     # xs = np.random.randint(0, W, size=num_points)
        #     # make grid to sample
        #     sep = 20
        #     grid_y, grid_x = np.mgrid[0:H:sep, 0:W:sep]
        #     ys, xs = grid_y.ravel(), grid_x.ravel()

        #     # 取出法向量 (x,y,z)
        #     gt_normals = gt_depth[ys, xs, :]
        #     pred_normals = pred_depth[ys, xs, :]

        #     # 归一化
        #     gt_normals = gt_normals / (np.linalg.norm(gt_normals, axis=1, keepdims=True) + 1e-8)
        #     pred_normals = pred_normals / (np.linalg.norm(pred_normals, axis=1, keepdims=True) + 1e-8)

        #     plt.figure(figsize=(18, 6))

        #     # -------- 左：GT 法线 --------
        #     plt.subplot(1, 3, 1)
        #     plt.imshow((gt_depth * 127.5 + 127.5).astype(np.uint8))  # normal map可视化到[0,255]
        #     plt.quiver(xs, ys, gt_normals[:, 0], -gt_normals[:, 1], color='r', scale=20, width=0.005)
        #     plt.title(f'GT Normals {i}')
        #     plt.axis('off')

        #     # -------- 中：Pred 法线 --------
        #     plt.subplot(1, 3, 2)
        #     plt.imshow((pred_depth * 127.5 + 127.5).astype(np.uint8))
        #     plt.quiver(xs, ys, pred_normals[:, 0], -pred_normals[:, 1], color='b', scale=20, width=0.005)
        #     plt.title(f'Pred Normals {i}')
        #     plt.axis('off')

        #     # -------- 右：GT depth + 两种箭头 --------
        #     plt.subplot(1, 3, 3)
        #     plt.imshow(gt_depth.astype(np.uint8))
        #     plt.quiver(xs, ys, gt_normals[:, 0], -gt_normals[:, 1], color='r', scale=20, width=0.005, label="GT")
        #     plt.quiver(xs, ys, pred_normals[:, 0], -pred_normals[:, 1], color='b', scale=20, width=0.005, label="Pred")
        #     plt.title(f'GT+Pred Normals {i}')
        #     plt.axis('off')
        #     plt.legend(loc="lower right")

        #     plt.tight_layout()
        #     plt.savefig(f'normals_compare_{i}.png', dpi=300)
        #     plt.close()


        try:
            mean_angular_error[i], median_angular_error[i], rmse_angular_error[i], sub5_error[i], sub7_5_error[i], sub11_25_error[i], sub22_5_error[i], sub30_error[i] = compute_normal_metrics(
                pred_depth, gt_depth)
        except Exception as e:
            print(f'Error computing metrics for sample {i}: {e}')
            continue

    # 过滤掉无效值
    valid_results = ~np.isnan(mean_angular_error) & ~np.isinf(mean_angular_error) 
    results = "{:7.3f}, {:7.3f}, {:7.3f}, {:7.3f}, {:7.3f}, {:7.3f}, {:7.3f}".format(
        mean_angular_error[valid_results].mean(), median_angular_error[valid_results].mean(), sub5_error[valid_results].mean(),
        sub7_5_error[valid_results].mean(), sub11_25_error[valid_results].mean(), sub22_5_error[valid_results].mean(), 
        sub30_error[valid_results].mean())
    print("{:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}".format(
        "mean", "median", "sub5", "sub7.5", "sub11.25", "sub22.5", "sub30")
    )
    print(results)
    
    print(f'Valid results: {valid_results.sum()}/{len(valid_results)}')
    return results
    # return silog, log10, abs_rel, sq_rel, rms, log_rms, d1, d2, d3



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='BTS TensorFlow implementation.', fromfile_prefix_chars='@')
    parser.convert_arg_line_to_args = convert_arg_line_to_args

    parser.add_argument('--pred_path',           type=str,   help='path to the prediction results in png', required=True)
    parser.add_argument('--gt_path',             type=str,   help='root path to the groundtruth data', required=False)
    parser.add_argument('--dataset',             type=str,   help='dataset to test on, nyu or kitti', default='nyu')
    parser.add_argument('--txt_file_list',      type=str,   help='text file containing list of files to evaluate', default=None)
    args = parser.parse_args()
    test(args)
