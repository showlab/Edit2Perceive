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

import os
import argparse
import fnmatch
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import struct
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'


def convert_arg_line_to_args(arg_line):
    for arg in arg_line.split():
        if not arg.strip():
            continue
        yield arg




def scale_and_shift_align(pred, gt, valid_mask):
    """
    使用最小二乘法对齐预测深度的scale和shift
    pred: 预测深度 (相对深度 0-255)
    gt: 真实深度 (绝对深度)
    valid_mask: 有效像素掩码
    """
    pred_valid = pred[valid_mask].flatten()
    gt_valid = gt[valid_mask].flatten()
    
    # 构建最小二乘法系统 Ax = b
    # 其中 A = [pred_valid, ones], x = [scale, shift], b = gt_valid
    A = np.vstack([pred_valid, np.ones(len(pred_valid))]).T
    scale, shift = np.linalg.lstsq(A, gt_valid, rcond=None)[0]
    
    # 应用scale和shift
    pred_aligned = pred * scale + shift
    
    return pred_aligned

def resize_depth_tensor(depth_img, target_height, target_width):
    """
    使用双线性插值调整深度图像大小
    """
    # 转换为tensor (1, 1, H, W)
    # depth_img = (depth_img - depth_img.min()) / (depth_img.max() - depth_img.min()) * np.float32(65535)
    # depth_img = depth_img.astype(np.uint16)
    depth_tensor = torch.from_numpy(depth_img).unsqueeze(0).unsqueeze(0).float()
    
    # 使用双线性插值调整大小
    resized_tensor = F.interpolate(
        depth_tensor, 
        size=(target_height, target_width), 
        mode='bilinear', 
        align_corners=True
    )
    
    # 转换回numpy
    resized_depth = resized_tensor.squeeze().numpy()

    return resized_depth

def read_depth(filename):
    with open(filename, 'rb') as f:
        tag = f.read(4)
        if tag != b'PIEH':
            raise ValueError("Invalid file format: expected 'PIEH' tag")
        
        width = struct.unpack('<I', f.read(4))[0]
        height = struct.unpack('<I', f.read(4))[0]
        
        depth_data = f.read(width * height * 4)
        if len(depth_data) != width * height * 4:
            raise ValueError("Incomplete depth data")
        
        # Convert the byte data to a list of floats
        depth_map = list(struct.unpack('<' + 'f' * (width * height), depth_data))
        # convert into a array
        depth_map = np.array(depth_map, dtype=np.float32).reshape((height, width)) 
    return depth_map

def read_depth_binary_file(file_path, image_width, image_height):
    """
    读取一个二进制格式的深度图文件，该文件实际上是由 float32 组成的，
    按行优先顺序存储，每个像素一个深度值（float32，4字节）。
    
    参数:
        file_path (str): 深度图文件的路径（例如：'depth.bin' 或 'depth.jpg'）
        image_width (int): 图像的宽度（像素数）
        image_height (int): 图像的高度（像素数）
        
    返回:
        numpy.ndarray: 一个形状为 (image_height, image_width) 的二维数组，
                      每个元素为一个 float32 的深度值，无深度处为 NaN。
    """
    # 每个像素 4 字节 (float32)
    try:
        bytes_per_pixel = 4
        total_pixels = image_width * image_height
        
        # 二进制文件总字节数
        expected_file_size = total_pixels * bytes_per_pixel
        
        # 以二进制方式读取文件
        with open(file_path, 'rb') as f:
            data = f.read()
        
        # 检查文件大小是否匹配预期
        if len(data) != expected_file_size:
            raise ValueError(f"文件大小不符合预期。期望 {expected_file_size} 字节，实际 {len(data)} 字节。"
                            f"请检查图像尺寸({image_width}x{image_height})是否正确。")
        
        # 将二进制数据解析为 float32 数组
        depth_values = np.frombuffer(data, dtype=np.float32)
        
        # 重塑为二维数组，形状为 (height, width)
        depth_map = depth_values.reshape((image_height, image_width))
        
        return depth_map
    except Exception as e:
        raise e

def load_image_rgb_or_grayscale(image_path):
    """
    加载图像，支持RGB和灰度图像，统一转换为numpy数组
    """
    try:
        if 'eth3d/depth' in image_path.lower(): # depth save as a 4bytes float32
            img_array = read_depth_binary_file(image_path, 6048, 4032)
        elif image_path.endswith('.dpt'):
            # 如果是dpt文件，直接读取
            img_array = read_depth(image_path)
        elif image_path.endswith('.npy'):
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
        return np.mean(img_array, axis=2) if len(img_array.shape) == 3 else img_array
        # return img_array[:,:,0] if len(img_array.shape) == 3 else img_array
    except:

        raise ValueError(f"Failed to load image from {image_path}. Ensure it is a valid image file or depth map.")
def compute_errors(gt, pred):
    thresh = np.maximum((gt / pred), (pred / gt))
    d1 = (thresh < 1.25).mean()
    d2 = (thresh < 1.25 ** 2).mean()
    d3 = (thresh < 1.25 ** 3).mean()

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)
    sq_rel = np.mean(((gt - pred)**2) / gt)

    err = np.log(pred) - np.log(gt)
    silog = np.sqrt(np.mean(err ** 2) - np.mean(err) ** 2) * 100

    err = np.abs(np.log10(pred) - np.log10(gt))
    log10 = np.mean(err)

    return silog, log10, abs_rel, sq_rel, rmse, rmse_log, d1, d2, d3


def test(args):
    global gt_depths, missing_ids, pred_filenames,gt_depths_mask
    gt_depths = []
    gt_depths_mask = []
    missing_ids = set()
    pred_filenames = []
    if "inverse" in args.pred_path:
        print('!!! Important: Inverse depth detected, will convert to depth during evaluation.')
    for root, dirnames, filenames in os.walk(args.pred_path):
        for pred_filename in fnmatch.filter(filenames, '*.png') + fnmatch.filter(filenames, '*.npy'):
            if 'cmap' in pred_filename or 'gt' in pred_filename:
                continue
            dirname = root.replace(args.pred_path, '')
            if dirname.startswith('/'):
                dirname = dirname[1:]
            pred_filenames.append(os.path.join(dirname, pred_filename))

    num_test_samples = len(pred_filenames)
    # print(f'Found {num_test_samples} prediction files')

    pred_depths = []
    if args.gt_path[-1]=='/':
        args.gt_path = args.gt_path[:-1]
    for i in range(num_test_samples):
        pred_depth_path = os.path.join(args.pred_path,pred_filenames[i])
        pred_depth = load_image_rgb_or_grayscale(pred_depth_path)
        
        if pred_depth is None:
            print('Missing: %s ' % pred_depth_path)
            missing_ids.add(i)
            continue

        # 预测图像是0-255的relative depth，先转换为float
        pred_depth = pred_depth.astype(np.float32)
        
        pred_depths.append(pred_depth)

    if args.dataset == 'kitti':
        for t_id in range(num_test_samples):
            if t_id in missing_ids:
                continue
                
            # 构建GT路径，保持与pred相同的目录结构
            pred_relative_path = pred_filenames[t_id]
            gt_depth_path = os.path.join(args.gt_path, pred_relative_path[11:]) # 去掉前面的20xx_xx_xx/
            gt_depth_path = gt_depth_path.replace("image_02/data","proj_depth/groundtruth/image_02").replace(".npy",".png")
            depth = cv2.imread(gt_depth_path, -1)
            if depth is None:
                print(f'Missing: {gt_depth_path} for pred file {pred_relative_path}')
                missing_ids.add(t_id)
                continue
            # depth = cv2.cvtColor(depth, cv2.COLOR_BGR2GRAY)
            depth = depth.astype(np.float32) / 256.0
            # print(f" depth shape: {depth.shape}")
            gt_depths.append(depth)
    elif args.dataset == 'nyu' or args.dataset == 'nyuv2':
        for t_id in range(num_test_samples):
            if t_id in missing_ids:
                continue
                
            # 构建GT路径，保持与pred相同的目录结构
            pred_relative_path = pred_filenames[t_id]
            gt_depth_path = os.path.join(args.gt_path, pred_relative_path)
            gt_depth_path = gt_depth_path.replace("rgb","depth").replace(".npy",".png")
            depth = cv2.imread(gt_depth_path, -1)
            if depth is None:
                print('Missing: %s ' % gt_depth_path)
                missing_ids.add(t_id)
                continue

            depth = depth.astype(np.float32) / 1000.0
            gt_depths.append(depth)
    elif args.dataset == 'Sintel':
        for t_id in range(num_test_samples):
            if t_id in missing_ids:
                continue
                
            # 构建GT路径，保持与pred相同的目录结构
            pred_relative_path = pred_filenames[t_id]
            gt_depth_path = os.path.join(args.gt_path, pred_relative_path)
            gt_depth_path = gt_depth_path.replace("final","depth").replace(".png",".dpt")
            depth = load_image_rgb_or_grayscale(gt_depth_path)
            if depth is None:
                print('Missing: %s ' % gt_depth_path)
                missing_ids.add(t_id)
                continue

            depth = depth.astype(np.float32) / 1000.0
            gt_depths.append(depth)
    elif args.dataset == 'diode':
        for t_id in range(num_test_samples):
            if t_id in missing_ids:
                continue
                
            # 构建GT路径，保持与pred相同的目录结构
            pred_relative_path = pred_filenames[t_id]
            gt_depth_path = os.path.join(args.gt_path, pred_relative_path)
            gt_depth_path = gt_depth_path.replace(".npy","_depth.npy")
            gt_depth_mask_path = gt_depth_path.replace("_depth.npy","_depth_mask.npy")
            # print(f"gt_depth_path = {gt_depth_path}")
            # print(f"gt_depth_mask_path = {gt_depth_mask_path}")
            depth = load_image_rgb_or_grayscale(gt_depth_path)
            depth_mask = load_image_rgb_or_grayscale(gt_depth_mask_path)
            if depth is None:
                print('Missing: %s ' % gt_depth_path)
                missing_ids.add(t_id)
                continue
            gt_depths.append(depth)
            gt_depths_mask.append(depth_mask)
    elif args.dataset == 'eth3d':
        for t_id in range(num_test_samples):
            if t_id in missing_ids:
                continue
            pred_relative_path = pred_filenames[t_id]
            # gt_depth_path = os.path.join(args.gt_path, pred_relative_path)
            # gt_depth_path = gt_depth_path.replace("rgb","depth").replace(".npy",".png")
            parts = pred_relative_path.split('/')
            assert parts[0]=='rgb'
            scene = parts[1]
            fixed_prefix = '/opt/liblibai-models/user-workspace2/users/syq/Depth_Post_Train/dataset/Eval/depth/ETH3D/'
            # 目标路径的模板：

            gt_depth_path = f"{args.gt_path}/depth/{scene}_dslr_depth/{scene}/ground_truth_depth/dslr_images/{parts[-1].replace('.npy','.JPG')}"
            # depth = cv2.imread(gt_depth_path, -1)
            depth = load_image_rgb_or_grayscale(gt_depth_path)

            if depth is None:
                print('Missing: %s ' % gt_depth_path)
                missing_ids.add(t_id)
                continue
            gt_depths.append(depth)
    elif args.dataset == 'scannet':
        for t_id in range(num_test_samples):
            if t_id in missing_ids:
                continue
                
            # 构建GT路径，保持与pred相同的目录结构
            pred_relative_path = pred_filenames[t_id]
            gt_depth_path = os.path.join(args.gt_path, pred_relative_path)
            gt_depth_path = gt_depth_path.replace("color","depth").replace(".npy",".png")
            depth = cv2.imread(gt_depth_path, -1)
            if depth is None:
                print('Missing: %s ' % gt_depth_path)
                missing_ids.add(t_id)
                continue

            depth = depth.astype(np.float32)/1000.0
            gt_depths.append(depth)
    else:
        raise ValueError('Unknown dataset: %s' % args.dataset)
    print(f'### Computing errors for {len(gt_depths)} files and {len(missing_ids)} missing' if not gt_depths_mask else f'Computing errors with masks for {len(gt_depths)} files and {len(missing_ids)} missing')

    result = eval(pred_depths,args)

    print('Done.')
    return result


def eval(pred_depths,args):
    num_samples = len(pred_depths)
    pred_depths_valid = []
    gt_depths_valid = []
    if args.using_pdf:
        pdf = np.load('depth_mapping_lookup_table.npz')
    # 收集有效的预测和GT深度
    gt_idx = 0
    for t_id in range(num_samples):
        if t_id in missing_ids:
            continue

        pred_depths_valid.append(pred_depths[t_id])
        gt_depths_valid.append(gt_depths[gt_idx])
        gt_idx += 1

    num_samples = len(pred_depths_valid)

    silog = np.zeros(num_samples, np.float32)
    log10 = np.zeros(num_samples, np.float32)
    rms = np.zeros(num_samples, np.float32)
    log_rms = np.zeros(num_samples, np.float32)
    abs_rel = np.zeros(num_samples, np.float32)
    sq_rel = np.zeros(num_samples, np.float32)
    d1 = np.zeros(num_samples, np.float32)
    d2 = np.zeros(num_samples, np.float32)
    d3 = np.zeros(num_samples, np.float32)
    
    for i in range(num_samples):
        gt_depth = gt_depths_valid[i]
        pred_depth = pred_depths_valid[i]

        # 1. 首先调整预测深度的大小以匹配GT
        if pred_depth.shape != gt_depth.shape:
            if args.do_kb_crop:
                target_h, target_w = 352, 1216
            else:
                target_h, target_w = gt_depth.shape[0], gt_depth.shape[1]
            pred_depth = resize_depth_tensor(pred_depth, target_h, target_w)
        gt_depth = gt_depth.copy()
        # 处理无效值
        gt_depth[np.isinf(gt_depth)] = 0
        gt_depth[np.isnan(gt_depth)] = 0
        
        pred_depth[np.isinf(pred_depth)] = 0
        pred_depth[np.isnan(pred_depth)] = 0

        # 创建有效掩码， 只评估 不为nan或者inf，且在深度范围内的像素
        valid_mask = np.logical_and(gt_depth > args.min_depth_eval, gt_depth < args.max_depth_eval)
        valid_mask = np.logical_and(valid_mask, ~np.isnan(gt_depth))
        valid_mask = np.logical_and(valid_mask, ~np.isinf(gt_depth))

        if gt_depths_mask:
            valid_mask = np.logical_and(valid_mask, gt_depths_mask[i] > 0)
        if args.dataset == 'nyu':
            _valid_mask = np.zeros_like(valid_mask)
            _valid_mask[45:471, 41:601] = 1
            valid_mask = np.logical_and(valid_mask, _valid_mask)
            del _valid_mask
        # 处理裁剪
        if args.do_kb_crop:
            height, width = gt_depth.shape
            top_margin = int(height - 352)
            left_margin = int((width - 1216) / 2)
            pred_depth_uncropped = np.zeros((height, width), dtype=np.float32)
            try:
                if abs(pred_depth.shape[0]-375) < 10:
                    pred_depth_uncropped[top_margin:top_margin + 352, left_margin:left_margin + 1216] = pred_depth[top_margin:top_margin + 352, left_margin:left_margin + 1216]
                    pred_depth = pred_depth_uncropped
                else:
                    pred_depth_uncropped[top_margin:top_margin + 352, left_margin:left_margin + 1216] = pred_depth
                    pred_depth = pred_depth_uncropped
            except Exception as e:
                print(f"Error in do_kb_crop for sample {i}: {e}")
                print(f"pred shape:{pred_depth.shape}, uncropped shape:{pred_depth_uncropped.shape}")
            _valid_mask = np.zeros_like(valid_mask)
            _valid_mask[top_margin:top_margin + 352, left_margin:left_margin + 1216] = valid_mask[top_margin:top_margin + 352, left_margin:left_margin + 1216]
            valid_mask = _valid_mask    
        if args.garg_crop or args.eigen_crop:
            gt_height, gt_width = gt_depth.shape
            eval_mask = np.zeros(valid_mask.shape)

            if args.garg_crop:
                eval_mask[int(0.40810811 * gt_height):int(0.99189189 * gt_height), int(0.03594771 * gt_width):int(0.96405229 * gt_width)] = 1

            elif args.eigen_crop:
                if args.dataset == 'kitti':
                    eval_mask[int(0.3324324 * gt_height):int(0.91351351 * gt_height), int(0.0359477 * gt_width):int(0.96405229 * gt_width)] = 1
                else:
                    eval_mask[45:471, 41:601] = 1

            valid_mask = np.logical_and(valid_mask, eval_mask)

        # 检查是否有足够的有效像素
        if valid_mask.sum() < 100:
            print(f'Warning: Sample {i} has very few valid pixels ({valid_mask.sum()})')
            continue

        # 2. 应用scale-and-shift对齐
        # print("original gt depth min:{:.4f} max:{:.4f} mean:{:.4f}".format(gt_depth[valid_mask].min(), gt_depth[valid_mask].max(), gt_depth[valid_mask].mean()))
        if getattr(args, 'using_log',None):
            gt_depth_cp = np.log(gt_depth+1e-6)
        elif getattr(args, 'using_sqrt_disp',None):
            gt_depth_cp = 1/np.sqrt(gt_depth+1e-8)
        elif getattr(args, 'using_disp',None):
            gt_depth_cp = 1/(gt_depth+1e-8)
        elif getattr(args, 'using_sqrt',None):
            gt_depth_cp = np.sqrt(gt_depth+1e-6)
        elif getattr(args, 'using_pdf',None):
            gt_depth_cp = np.interp(gt_depth, pdf['bins'], pdf['y_map'])
        else:
            gt_depth_cp = gt_depth.copy()
        pred_depth_aligned = scale_and_shift_align(pred_depth, gt_depth_cp, valid_mask)
        if getattr(args, 'using_log',None):
            pred_depth_aligned = np.exp(pred_depth_aligned)
        elif getattr(args, 'using_sqrt_disp',None):
            pred_depth_aligned = 1/(pred_depth_aligned**2)
        elif getattr(args, 'using_disp',None):
            pred_depth_aligned = 1/pred_depth_aligned
        elif getattr(args, 'using_sqrt',None):
            pred_depth_aligned = np.power(pred_depth_aligned, 2)
        elif getattr(args, 'using_pdf',None):
            pred_depth_aligned = np.interp(pred_depth_aligned, pdf['y_map'], pdf['bins'])
        pred_depth_aligned = np.clip(pred_depth_aligned, args.min_depth_eval, args.max_depth_eval)

        # 计算误差
        try:
            silog[i], log10[i], abs_rel[i], sq_rel[i], rms[i], log_rms[i], d1[i], d2[i], d3[i] = compute_errors(
                gt_depth[valid_mask], pred_depth_aligned[valid_mask])
        except Exception as e:
            print(f'Error computing metrics for sample {i}: {e}')
            continue

    # 过滤掉无效值
    valid_results = ~np.isnan(silog) & ~np.isinf(silog) & (silog != 0)
    results = "{:7.5f}, {:7.5f}, {:7.5f}, {:7.5f}, {:7.5f}, {:7.5f}, {:7.5f}, {:7.5f}, {:7.5f}".format(
            d1[valid_results].mean(), d2[valid_results].mean(), d3[valid_results].mean(),
            abs_rel[valid_results].mean(), sq_rel[valid_results].mean(), rms[valid_results].mean(), 
            log_rms[valid_results].mean(), silog[valid_results].mean(), log10[valid_results].mean())
    if not args.no_verbose:
        print("{:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}".format(
            'd1', 'd2', 'd3', 'AbsRel', 'SqRel', 'RMSE', 'RMSElog', 'SILog', 'log10'))
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
    parser.add_argument('--eigen_crop',                      help='if set, crops according to Eigen NIPS14', action='store_true')
    parser.add_argument('--garg_crop',                       help='if set, crops according to Garg  ECCV16', action='store_true')
    parser.add_argument('--min_depth_eval',      type=float, help='minimum depth for evaluation', default=1e-3)
    parser.add_argument('--max_depth_eval',      type=float, help='maximum depth for evaluation', default=80)
    parser.add_argument('--do_kb_crop',                      help='if set, crop input images as kitti benchmark images', action='store_true')
    parser.add_argument('--no_verbose', default=False, action='store_true', help='if set, do not print out per image results')
    parser.add_argument('--using_log', default=False, action='store_true', help='if set, use log depth for eval')
    parser.add_argument('--using_disp', default=False, action='store_true', help='if set, use disparity (1/depth) for eval')
    parser.add_argument('--using_sqrt', default=False, action='store_true', help='if set, use sqrt depth for eval')
    parser.add_argument('--using_pdf', default=False, action='store_true', help='if set, use pdf for eval')
    args = parser.parse_args()
    test(args)
    # load_image_rgb_or_grayscale("/opt/liblibai-models/user-workspace2/users/syq/Depth_Post_Train/dataset/Eval/depth/ETH3D/depth/kicker_dslr_depth/kicker/ground_truth_depth/dslr_images/DSC_6493.JPG")