# from diffsynth.utils.metric import compute_matting_metrics
# from PIL import Image
# import numpy as np
# import os
# import argparse
# import fnmatch
# import torch
# import torch.nn.functional as F
# import matplotlib.pyplot as plt
# from tqdm import trange

# def resize_tensor(input_tensor, target_height, target_width):
#     """
#     使用双线性插值调整深度图像大小
#     """
#     # 多通道resize
#     input_tensor = torch.from_numpy(input_tensor).unsqueeze(0).unsqueeze(0).float()
#     # 使用双线性插值调整大小
#     resized_tensor = F.interpolate(
#         input_tensor, 
#         size=(target_height, target_width), 
#         mode='bilinear', 
#         align_corners=True
#     )
#     # 转换回numpy
#     resized = resized_tensor.squeeze().numpy()
#     return resized

# def load_image_rgb_or_grayscale(image_path):
#     """
#     加载图像，支持RGB和灰度图像，统一转换为numpy数组
#     """
#     try:
#         if image_path.endswith('.npy'):
#             # 如果是npy文件，直接读取
#             img_array = np.load(image_path)
#         else:
#             # 首先尝试用PIL加载，可以更好地处理不同格式
#             img = Image.open(image_path)
#             img_array = np.array(img)
            
#             # 如果是RGBA图像，取前3通道，否则RGB或者Gray则不处理
#             if len(img_array.shape) == 3 and img_array.shape[2] == 4:  # RGBA
#                 img_array = img_array[:, :, :3]  # 去掉Alpha通道
#             # 对3通道取均值返回
#         return np.mean(img_array,axis=2) if len(img_array.shape) == 3 else img_array
#     except:
#         raise ValueError(f"Failed to load image from {image_path}. Ensure it is a valid image file or depth map.")

# def test(args):
#     global gt_depths, missing_ids, pred_filenames,gt_depths_mask
#     gt_depths = []
#     gt_depths_mask = []
#     missing_ids = set()
#     pred_depths = []
#     if args.dataset == "comp":
#         with open("/mnt/nfs/workspace/syq/Marigold/data_split/comp_matting/filenames_test.txt","r") as f:
#             test_filenames = [line.strip("\t") for line in f.readlines()]
#         # each line have 4 items: merged, trimap, fg, alpha
#         # here alpha is the gt depth (indeed not depth but a matting alpha mask, we use it for same name)
#         if "full" not in args.pred_path and "result" not in args.pred_path:
#             test_filenames = test_filenames[:10]
#     elif args.dataset == "aim":
#         with open("/mnt/nfs/workspace/syq/Marigold/data_split/AIM_matting/filenames_val.txt","r") as f:
#             test_filenames = [line.strip() for line in f.readlines()]
#     elif args.dataset == "p3m":
#         with open("/mnt/nfs/workspace/syq/Marigold/data_split/P3M_matting/filenames_val_P.txt","r") as f:
#             test_filenames = [line.strip() for line in f.readlines()]
#         if "full" not in args.pred_path and "result" not in args.pred_path:
#             test_filenames = test_filenames[:10]
#     elif args.dataset == "p3m-np":
#         with open("/mnt/nfs/workspace/syq/Marigold/data_split/P3M_matting/filenames_val_NP.txt","r") as f:
#             test_filenames = [line.strip() for line in f.readlines()]
#             if "full" not in args.pred_path and "result" not in args.pred_path:
#                 test_filenames = test_filenames[:10]
#     elif args.dataset == "am":
#         with open("/mnt/nfs/workspace/syq/Marigold/data_split/AM_matting/filenames_val.txt","r") as f:
#             test_filenames = [line.strip() for line in f.readlines()]
#         if "full" not in args.pred_path and "result" not in args.pred_path:
#             test_filenames = test_filenames[:10]
#     # if args.max_samples is not None:
#     if getattr(args, 'max_samples', None) is not None:
#         test_filenames = test_filenames[:args.max_samples]
#     for t_id, line in enumerate(test_filenames):
#         items = line.split()
#         if len(items) != 3:
#             merged_path, trimap_path, _, alpha_path = items
#         else:
#             merged_path, trimap_path, alpha_path = items
#         pred = load_image_rgb_or_grayscale(os.path.join(args.pred_path, merged_path.replace(".jpg",".npy").replace(".png",".npy")))
#         pred_depths.append(pred)
#         trimap = load_image_rgb_or_grayscale(os.path.join(args.gt_path, trimap_path))
#         alpha = load_image_rgb_or_grayscale(os.path.join(args.gt_path, alpha_path))
#         gt_depths.append(alpha)
#         gt_depths_mask.append(trimap)

#     print(f'### Computing errors for {len(gt_depths)} files with {len(missing_ids)} missing' if not gt_depths_mask else 'Computing errors with masks')

#     results = eval(pred_depths,args)

#     print('Done.')
#     return results

# def eval(pred_depths,args):
#     num_samples = len(pred_depths)
#     pred_depths_valid = []
#     gt_depths_valid = []

#     # 收集有效的预测和GT深度
#     gt_idx = 0
#     for t_id in range(num_samples):
#         if t_id in missing_ids:
#             continue

#         pred_depths_valid.append(pred_depths[t_id])
#         gt_depths_valid.append(gt_depths[gt_idx])
#         gt_idx += 1

#     num_samples = len(pred_depths_valid)

#     sad_error = np.zeros(num_samples, np.float32)
#     mse_error = np.zeros(num_samples, np.float32)
#     grad_error = np.zeros(num_samples, np.float32)
#     conn_error = np.zeros(num_samples, np.float32)
#     mad_error = np.zeros(num_samples, np.float32)
#     valid_results = np.zeros(num_samples, np.bool_)
#     for i in trange(num_samples):
#         gt_depth = gt_depths_valid[i]
#         gt_depth[np.isinf(gt_depth)] = 0
#         gt_depth[np.isnan(gt_depth)] = 0
#         pred_depth = pred_depths_valid[i]
#         pred_depth[np.isinf(pred_depth)] = 0
#         pred_depth[np.isnan(pred_depth)] = 0
#         trimap = gt_depths_mask[i]
#         if pred_depth.shape != gt_depth.shape:
#             pred_depth = resize_tensor(pred_depth, gt_depth.shape[0], gt_depth.shape[1])
#         # pred_depth[trimap==0] = 0
#         # pred_depth[trimap==255] = 1
#         pred_depth = pred_depth>0.5
#         gt_depth = gt_depth.astype(np.float32)/255.0
#         # try:
#         #     sad, mse, grad, conn = compute_matting_metrics(pred_depth, gt_depth, trimap)
#         #     sad_error[i] = sad
#         #     mse_error[i] = mse
#         #     grad_error[i] = grad
#         #     conn_error[i] = conn
#         try:
#             mse, mad, sad, grad, conn = compute_matting_metrics(pred_depth, gt_depth, trimap,whole=True)
#             sad_error[i] = sad
#             mse_error[i] = mse
#             mad_error[i] = mad
#             grad_error[i] = grad
#             conn_error[i] = conn
#             valid_results[i] = (sad >= 0) & (mse >= 0) & (mad >= 0) & (conn >= 0)

#         except Exception as e:
#             print(f'Error computing metrics for sample {i}: {e}')
#             continue


#     results = "{:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}".format(
#         mse_error.mean(), mad_error.mean(), sad_error.mean(), grad_error.mean(), conn_error.mean()
#     )
#     print("{:>7}, {:>7}, {:>7}, {:>7}, {:>7}".format(
#         'MSE', 'MAD', 'SAD', 'Grad', 'Conn')
#     )
#     print(results)
    
#     print(f'Valid results: {valid_results.sum()}/{len(valid_results)}')
#     return results
#     # return silog, log10, abs_rel, sq_rel, rms, log_rms, d1, d2, d3



# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(description='BTS TensorFlow implementation.', fromfile_prefix_chars='@')

#     parser.add_argument('--pred_path',           type=str,   help='path to the prediction results in png')
#     parser.add_argument('--gt_path',             type=str,   help='root path to the groundtruth data' )
#     parser.add_argument('--dataset',             type=str,   help='dataset to test on, nyu or kitti', default='p3m')
#     parser.add_argument('--max_samples',        type=int,    help='maximum number of samples to test', default=None)
#     args = parser.parse_args()
#     test(args)

from utils.metric import compute_matting_metrics
from PIL import Image
import numpy as np
import os
import argparse
import torch
import torch.nn.functional as F
from tqdm import tqdm # 使用tqdm来显示进度条，对并行任务尤其友好
import multiprocessing as mp # 核心库

# Helper functions (resize_tensor, load_image_rgb_or_grayscale) 保持不变
def resize_tensor(input_tensor, target_height, target_width):
    """
    使用双线性插值调整深度图像大小
    """
    input_tensor = torch.from_numpy(input_tensor).unsqueeze(0).unsqueeze(0).float()
    resized_tensor = F.interpolate(
        input_tensor, 
        size=(target_height, target_width), 
        mode='bilinear', 
        align_corners=True
    )
    return resized_tensor.squeeze().numpy()

def load_image_rgb_or_grayscale(image_path):
    """
    加载图像，支持RGB和灰度图像，统一转换为numpy数组
    """
    try:
        if image_path.endswith('.npy'):
            img_array = np.load(image_path)
        else:
            img = Image.open(image_path)
            img_array = np.array(img)
            if len(img_array.shape) == 3 and img_array.shape[2] == 4:
                img_array = img_array[:, :, :3]
        return np.mean(img_array, axis=2) if len(img_array.shape) == 3 else img_array
    except Exception as e:
        # 增加错误日志，方便调试
        print(f"Warning: Failed to load image from {image_path}. Error: {e}")
        return None

# ------------------- 新增：独立的工作函数 -------------------
# 这个函数包含了处理单个样本的所有逻辑，它将在一个独立的进程中被调用。
def process_sample(task_args):
    """
    处理单个样本：加载数据、预处理、计算指标。
    Args:
        task_args (tuple): 包含单个任务所需所有路径的元组 (pred_path, alpha_path, trimap_path)
    Returns:
        tuple: 包含 (mse, mad, sad, grad, conn) 的元组，如果出错则返回五个 np.nan
    """
    pred_path, alpha_path, trimap_path = task_args

    try:
        # 1. 加载数据
        pred_depth = load_image_rgb_or_grayscale(pred_path)
        alpha = load_image_rgb_or_grayscale(alpha_path)
        trimap = load_image_rgb_or_grayscale(trimap_path)

        # 如果任何一个文件加载失败，则跳过
        if pred_depth is None or alpha is None or trimap is None:
            return (np.nan, np.nan, np.nan, np.nan, np.nan)

        # 2. 预处理 (与原eval函数中的逻辑相同)
        gt_depth = alpha
        gt_depth[np.isinf(gt_depth)] = 0
        gt_depth[np.isnan(gt_depth)] = 0
        pred_depth[np.isinf(pred_depth)] = 0
        pred_depth[np.isnan(pred_depth)] = 0

        if pred_depth.shape != gt_depth.shape:
            pred_depth = resize_tensor(pred_depth, gt_depth.shape[0], gt_depth.shape[1])
        
        # pred_depth = pred_depth > 0.5
        gt_depth = gt_depth.astype(np.float32) / 255.0

        # 3. 计算指标
        mse, mad, sad, grad, conn = compute_matting_metrics(pred_depth, gt_depth, trimap, whole=True)
        
        # 检查结果是否有效
        if not all(v >= 0 for v in [mse, mad, sad, grad, conn]):
             return (np.nan, np.nan, np.nan, np.nan, np.nan)

        return (mse, mad, sad, grad, conn)

    except Exception as e:
        print(f'Error processing {os.path.basename(pred_path)}: {e}')
        # 返回NaN，方便后续统计时忽略错误样本
        return (np.nan, np.nan, np.nan, np.nan, np.nan)

# ------------------- 重构后的 test 函数 (接口不变) -------------------
def test(args):
    # 1. 准备任务列表 (与原代码相同)
    # 这部分逻辑不变，只是现在我们不加载图像，而是收集文件路径来创建任务
    dataset_paths = {
        "comp": "/mnt/nfs/workspace/syq/Marigold/data_split/comp_matting/filenames_test.txt",
        "aim": "/mnt/nfs/workspace/syq/Marigold/data_split/AIM_matting/filenames_val.txt",
        "p3m": "/mnt/nfs/workspace/syq/Marigold/data_split/P3M_matting/filenames_val_P.txt",
        "p3m-np": "/mnt/nfs/workspace/syq/Marigold/data_split/P3M_matting/filenames_val_NP.txt",
        "am": "/mnt/nfs/workspace/syq/Marigold/data_split/AM_matting/filenames_val.txt",
    }
    
    filenames_path = dataset_paths.get(args.dataset)
    if not filenames_path:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    with open(filenames_path, "r") as f:
        test_filenames = [line.strip() for line in f.readlines()]
    
    # 在非full模式下截断文件名列表
    if "full" not in args.pred_path and "result" not in args.pred_path and args.dataset in ["comp", "p3m", "p3m-np", "am"]:
        test_filenames = test_filenames[:10]

    if getattr(args, 'max_samples', None) is not None:
        test_filenames = test_filenames[:args.max_samples]

    # 2. 创建任务列表，每个任务是一个包含所需文件路径的元组
    tasks = []
    for line in test_filenames:
        items = line.split()
        # 兼容不同格式的 filenames.txt
        merged_path, trimap_path, alpha_path = items[0], items[1], items[-1]
        
        pred_full_path = os.path.join(args.pred_path, merged_path.replace(".jpg",".npy").replace(".png",".npy"))
        alpha_full_path = os.path.join(args.gt_path, alpha_path)
        trimap_full_path = os.path.join(args.gt_path, trimap_path)
        
        tasks.append((pred_full_path, alpha_full_path, trimap_full_path))

    print(f'### Starting evaluation for {len(tasks)} files on dataset "{args.dataset}"...')
    mp_context = mp.get_context('spawn')
    # 3. 使用多进程池执行任务
    # os.cpu_count() 可以获取机器的CPU核心数，自动利用所有资源
    # `with`语句可以确保进程池在使用后被正确关闭
    results_list = []
    with mp_context.Pool(processes=32) as pool:
        # pool.imap_unordered 比 map 更高效，因为它不保证返回结果的顺序
        # 使用tqdm来包装，可以实时看到处理进度
        results_iterator = pool.imap_unordered(process_sample, tasks)
        
        for result in tqdm(results_iterator, total=len(tasks), desc="Processing samples"):
            results_list.append(result)

    # 4. 汇总结果
    # 将结果列表转换为numpy数组，方便进行列操作
    results_array = np.array(results_list)
    
    # 使用 np.nanmean 来计算平均值，它会自动忽略所有NaN值（即处理失败的样本）
    # 这比手动过滤更简洁、更安全
    mean_mse = np.nanmean(results_array[:, 0])
    mean_mad = np.nanmean(results_array[:, 1])
    mean_sad = np.nanmean(results_array[:, 2])
    mean_grad = np.nanmean(results_array[:, 3])
    mean_conn = np.nanmean(results_array[:, 4])

    # 统计有效结果的数量
    valid_results_count = np.sum(~np.isnan(results_array[:, 0]))

    # 5. 打印结果 (与原代码格式一致)
    final_results_str = "{:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}".format(
        mean_mse, mean_mad, mean_sad, mean_grad, mean_conn
    )
    
    print('Done.')
    print(f'Valid results: {valid_results_count}/{len(tasks)}')
    print("{:>7}, {:>7}, {:>7}, {:>7}, {:>7}".format('MSE', 'MAD', 'SAD', 'Grad', 'Conn'))
    print(final_results_str)
    
    return final_results_str

# 主程序入口保持不变
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='BTS TensorFlow implementation.', fromfile_prefix_chars='@')
    parser.add_argument('--pred_path',           type=str,   help='path to the prediction results in png')
    parser.add_argument('--gt_path',             type=str,   help='root path to the groundtruth data' )
    parser.add_argument('--dataset',             type=str,   help='dataset to test on, e.g., p3m, comp, etc.', default='p3m')
    parser.add_argument('--max_samples',        type=int,    help='maximum number of samples to test', default=None)
    args = parser.parse_args()
    test(args)