import torch
import numpy as np
import torch.nn.functional as F
import math
import torch.nn as nn

def _compute_sobel_gradients_chunked(image, chunk_size=4):
    """
    Computes Sobel gradients for a batch of images in chunks to save memory.
    
    Args:
        image: Tensor of shape [B, H, W]
        chunk_size: The size of each chunk to process.
    
    Returns:
        tuple: (grad_x, grad_y), each of shape [B, H, W]
    """
    # Sobel kernels
    sobel_x_kernel = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                                  dtype=image.dtype, device=image.device).view(1, 1, 3, 3)
    sobel_y_kernel = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                                  dtype=image.dtype, device=image.device).view(1, 1, 3, 3)

    batch_size = image.shape[0]
    # 如果 batch_size 小于等于 chunk_size，就不需要分块
    if batch_size <= chunk_size:
        image_unsqueezed = image.unsqueeze(1)
        grad_x = F.conv2d(image_unsqueezed, sobel_x_kernel, padding=1).squeeze(1)
        grad_y = F.conv2d(image_unsqueezed, sobel_y_kernel, padding=1).squeeze(1)
        return grad_x, grad_y

    # --- 分块计算 ---
    grads_x_list = []
    grads_y_list = []
    # 使用 torch.no_grad() 上下文可以进一步节省显存，因为我们不需要为卷积操作本身计算梯度
    # Sobel核是固定的，不需要梯度。输入image的梯度会通过torch自动追踪。
    with torch.no_grad():
        for i in range(0, batch_size, chunk_size):
            chunk = image[i:i + chunk_size]
            chunk_unsqueezed = chunk.unsqueeze(1)
            
            # 对每个块进行卷积
            chunk_grad_x = F.conv2d(chunk_unsqueezed, sobel_x_kernel, padding=1)
            chunk_grad_y = F.conv2d(chunk_unsqueezed, sobel_y_kernel, padding=1)
            
            grads_x_list.append(chunk_grad_x)
            grads_y_list.append(chunk_grad_y)
    
    # 合并结果
    grad_x = torch.cat(grads_x_list, dim=0).squeeze(1)
    grad_y = torch.cat(grads_y_list, dim=0).squeeze(1)
    
    return grad_x, grad_y

# def get_cycle_consistency_depth_loss(pred_depth, gt_depth, gt_mask=None, depth_normalization="log",eps=1e-6):
#     """
#     Cycle consistency depth loss based on AbsRel metric (Optimized batch processing version)
    
#     Args:
#         pred_depth: [B,C,H,W] or [B,1,H,W] - predicted depth maps
#         gt_depth: [B,C,H,W] or [B,1,H,W] - ground truth depth maps
#         gt_mask: [B,H,W] or [B,1,H,W] - mask for valid GT positions (optional)
#         eps: float - small value to avoid division by zero
#         depth_normalization: str - depth normalization method
    
#     Returns:
#         torch.Tensor - scalar loss value for backpropagation
#     """
    
#     # Ensure consistent shape -> [B, H, W]
#     if pred_depth.dim() == 4 and pred_depth.shape[1] != 1:
#         pred_depth = torch.mean(pred_depth, dim=1, keepdim=False)
#     elif pred_depth.dim() == 4:
#         pred_depth = pred_depth.squeeze(1)
    
#     if gt_depth.dim() == 4 and gt_depth.shape[1] != 1:
#         gt_depth = torch.mean(gt_depth, dim=1, keepdim=False)
#     elif gt_depth.dim() == 4:
#         gt_depth = gt_depth.squeeze(1)
    
#     # Handle shape mismatch by resizing prediction to match GT
#     if pred_depth.shape != gt_depth.shape:
#         pred_depth = F.interpolate(pred_depth.unsqueeze(1), size=gt_depth.shape[-2:], 
#                                  mode='bilinear', align_corners=False).squeeze(1)
    
#     # Handle invalid values in GT and prediction (batch processing)
#     gt_depth = torch.where(torch.isinf(gt_depth), torch.zeros_like(gt_depth), gt_depth)
#     gt_depth = torch.where(torch.isnan(gt_depth), torch.zeros_like(gt_depth), gt_depth)
    
#     pred_depth = torch.where(torch.isinf(pred_depth), torch.zeros_like(pred_depth), pred_depth)
#     pred_depth = torch.where(torch.isnan(pred_depth), torch.zeros_like(pred_depth), pred_depth)
    
#     # Create valid mask
#     if gt_mask is not None:
#         if gt_mask.dim() == 4:
#             gt_mask = gt_mask.squeeze(1)  # [B, H, W]
#         valid_mask = gt_mask.bool()
#     else:
#         valid_mask = torch.ones_like(gt_depth, dtype=torch.bool, device=gt_depth.device)
    
#     # Clamp depths to valid range
#     pred_depth = torch.clamp(pred_depth, -1.0, 1.0)
#     gt_depth = torch.clamp(gt_depth, -1.0, 1.0)
    
#     # Apply depth normalization (batch processing)
#     if depth_normalization == "log":
#         pred_depth = torch.exp(pred_depth)
#         gt_depth = torch.exp(gt_depth)
#     elif depth_normalization == "sqrt":
#         pred_depth = pred_depth**2
#         gt_depth = gt_depth**2
#     elif depth_normalization == "disp":
#         pred_depth = 1.0/(pred_depth + eps)
#         gt_depth = 1.0/(gt_depth + eps)
#     elif depth_normalization == "sqrt_disp":
#         pred_depth = 1.0/(pred_depth**2 + eps)
#         gt_depth = 1.0/(gt_depth**2 + eps)
#     elif depth_normalization == "uniform":
#         pass # already in the same scale as real relative depth
#     else:
#         raise ValueError(f"Unknown depth normalization: {depth_normalization}")
    
#     # Compute AbsRel metric for each sample in the batch
#     pred_depth = torch.log(pred_depth + eps)  # [B, H, W]
#     gt_depth = torch.log(gt_depth + eps)      # [B, H, W
#     abs_diff = torch.abs(pred_depth - gt_depth)  # [B, H, W]
#     rel_diff = abs_diff
#     # rel_diff = abs_diff / (gt_depth + eps)       # [B, H, W], add eps to avoid division by zero
    
#     # Apply mask and compute mean for each sample
#     masked_rel_diff = rel_diff * valid_mask.float()  # [B, H, W]
    
#     # Sum over spatial dimensions and divide by number of valid pixels per sample
#     valid_pixel_count = valid_mask.sum(dim=[1, 2]).float()  # [B]
#     sample_losses = masked_rel_diff.sum(dim=[1, 2])         # [B]
    
#     # Avoid division by zero for samples with no valid pixels
#     valid_samples_mask = valid_pixel_count > 0
#     sample_losses = sample_losses / torch.clamp(valid_pixel_count, min=1.0)
    
#     # Only consider samples with valid pixels
#     if valid_samples_mask.any():
#         final_loss = sample_losses[valid_samples_mask].mean()
#     else:
#         # If no valid samples, return zero loss
#         final_loss = torch.tensor(0.0, device=pred_depth.device, requires_grad=True)
    
#     return final_loss

class ScaleAndShiftInvariantLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.name = "SSILoss"
    def forward(self, prediction, target, mask,depth_normalization="log"):
        # if mask.ndim ==3:
        #     mask = mask.unsqueeze(1)  # [B,1,H,W]
        if prediction.ndim == 4:
            prediction = torch.mean(prediction, dim=1, keepdim=False) # [B,H,W]
        if target.ndim ==4:
            target = torch.mean(target, dim=1, keepdim=False) # [B,H,W]
        mask = mask.bool()
        with torch.autocast(device_type='cuda', enabled=False):
            prediction = prediction.float()
            target     = target.float()
            if depth_normalization == "log":
                target_cp = torch.log(target).requires_grad_(False)
            elif depth_normalization == "sqrt":
                target_cp = torch.sqrt(target).requires_grad_(False)
            scale, shift = compute_scale_and_shift_masked(prediction, target_cp, mask)
            del target_cp
            scaled_prediction = scale.view(-1, 1, 1) * prediction + shift.view(-1, 1, 1)
            loss = nn.functional.l1_loss(scaled_prediction[mask], target[mask])
        return loss
    
def compute_scale_and_shift_masked(prediction, target, mask):
    # system matrix: A = [[a_00, a_01], [a_10, a_11]]
    a_00 = torch.sum(mask * prediction * prediction, (1, 2))
    a_01 = torch.sum(mask * prediction, (1, 2))
    a_11 = torch.sum(mask, (1, 2))
    # right hand side: b = [b_0, b_1]
    b_0 = torch.sum(mask * prediction * target, (1, 2))
    b_1 = torch.sum(mask * target, (1, 2))
    # solution: x = A^-1 . b = [[a_11, -a_01], [-a_10, a_00]] / (a_00 * a_11 - a_01 * a_10) . b
    x_0 = torch.zeros_like(b_0)
    x_1 = torch.zeros_like(b_1)
    det = a_00 * a_11 - a_01 * a_01
    # A needs to be a positive definite matrix.
    valid = det > 0 #1e-3
    x_0[valid] = (a_11[valid] * b_0[valid] - a_01[valid] * b_1[valid]) / det[valid]
    x_1[valid] = (-a_01[valid] * b_0[valid] + a_00[valid] * b_1[valid]) / det[valid]
    return x_0, x_1

def get_cycle_consistency_depth_loss(pred, gt, mask=None, depth_normalization="log",eps=1e-8,
                   alpha=0.85, # SILog 损失的权重
                   beta=1.0,   # 梯度损失的权重
                   gamma=0.0,
                   per_sample_weights=None): # 可选的法线一致性损失的权重
    """
    一个专注于几何结构的、稳定版的深度损失函数。
    它结合了尺度不变损失(SILog)和局部结构损失(梯度匹配)。
    Args:
        pred: [B,3,H,W] 或 [B,H,W,3], 范围 [-1, 1]
        gt:   [B,3,H,W] 或 [B,H,W,3], 范围 [-1, 1]
        mask: None 或 [B,H,W] / [B,1,H,W] / [B,H,W,1]
        eps:  数值稳定项
        alpha: SILog 损失中的方差项权重 (通常 0.5 或 0.85)
        beta: 梯度损失的权重
        gamma: 法线一致性损失的权重 (如果>0，会计算法线损失)
    Returns:
        单标量 tensor（可反传）
    """
    # --- 形状、device、dtype 规范化 ---
    if pred.dim() == 4 and pred.shape[1] != 3 and pred.shape[-1] == 3:
        pred = pred.permute(0, 3, 1, 2)
    if gt.dim() == 4 and gt.shape[1] != 3 and gt.shape[-1] == 3:
        gt = gt.permute(0, 3, 1, 2)

    B, C, H, W = pred.shape
    device = pred.device
    dtype = pred.dtype
    # if per_sample_weights is None:
    #     per_sample_weights = torch.ones((B,), device=device, dtype=dtype)
    # --- 1. 预处理 ---
    # 取3通道均值得到单通道深度图
    pred = pred.mean(dim=1,keepdim=True) # [B,1,H,W]
    gt   = gt.mean(dim=1,keepdim=True)   # [B,1,H,W]

    # 将 [-1, 1] 范围映射到正数范围 (e.g., [eps, 1+eps])，为 log 做准备
    # 这一步非常重要，因为它将归一化的值转回到了一个类似深度的正数空间
    pred = (pred.clamp(min=-1, max=1) + 1.0) / 2.0 + eps
    gt   = (gt.clamp(min=-1, max=1) + 1.0) / 2.0 + eps
    # if depth_normalization == "log":
    #     pred = torch.exp(pred)
    #     gt = torch.exp(gt)
    # elif depth_normalization == "sqrt":
    #     pred = pred**2
    #     gt = gt**2
    # elif depth_normalization == "disp":
    #     pred = 1.0/(pred + eps)
    #     gt = 1.0/(gt + eps)
    # elif depth_normalization == "sqrt_disp":
    #     pred = 1.0/(pred**2 + eps)
    #     gt = 1.0/(gt**2 + eps)
    # else:
    #     # uniform 
    #     pass # already in the same scale as real relative depth
    # --- mask 处理 ---
    if mask is None:
        # 如果没有mask，创建一个全为True的mask
        mask = torch.ones_like(pred, dtype=torch.bool, device=device)
    else:
        # 确保mask是 [B,H,W] 的 bool tensor
        if mask.ndim == 4:
            mask = torch.mean(mask, dim=1,keepdim=True).bool()  # [B,1,H,W]
        elif mask.ndim ==3:
            mask = mask.unsqueeze(1).bool()  # [B,1,H,W]
    mask = mask.bool().to(device)
    # 检查有效像素数量
    total_valid = mask.sum()
    if total_valid.item() == 0:
        return torch.zeros((), device=device, dtype=dtype, requires_grad=True)

    # --- 2. 计算复合损失 ---

    # --- Component A: Scale-Invariant Logarithmic (SILog) Loss ---
    # 这是点对点损失的鲁棒版本，对尺度不敏感
    log_diff = torch.log(1+pred[mask]) - torch.log(1+gt[mask]) # B,N

    # # SILog 公式: E[d^2] - alpha * (E[d])^2
    silog_term1 = torch.mean(log_diff ** 2)
    silog_term2 = (torch.mean(log_diff)) ** 2
    loss_silog = silog_term1 - alpha * silog_term2
    loss_silog = (loss_silog).mean()
    # --- Component B: Gradient Matching Loss ---
    # 这是结构损失，确保表面的局部坡度一致
    # 使用简单的 sobel 算子来计算梯度
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], device=device, dtype=dtype).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], device=device, dtype=dtype).view(1, 1, 3, 3)

    pred_grad_x = F.conv2d(pred, sobel_x, padding=1)
    pred_grad_y = F.conv2d(pred, sobel_y, padding=1)
    gt_grad_x = F.conv2d(gt, sobel_x, padding=1)
    gt_grad_y = F.conv2d(gt, sobel_y, padding=1)

    # 计算梯度差异的 L1 损失
    grad_loss_x = torch.abs(pred_grad_x - gt_grad_x)
    grad_loss_y = torch.abs(pred_grad_y - gt_grad_y)

    # 只在 mask 区域计算损失
    # 稍微腐蚀一下mask，因为边界处的梯度可能不准确
    mask_eroded = F.max_pool2d(mask.float(), kernel_size=3, stride=1, padding=1).bool()
    loss_grad = (grad_loss_x[mask_eroded].mean() + grad_loss_y[mask_eroded].mean())
    # loss_grad = ((grad_loss_x[mask_eroded].view(B, -1).mean(dim=1) + grad_loss_y[mask_eroded].view(B, -1).mean(dim=1)) * per_sample_weights).mean()

    # --- Component C (Optional): Normal Consistency Loss ---
    # 如果 gamma > 0，则计算从深度图重建的法线之间的一致性损失
    loss_normal = torch.tensor(0.0, device=device, dtype=dtype)
    if gamma > 0:
        # 从深度图计算法线 (使用梯度)
        # 法线 n = [-gx, -gy, 1]，然后归一化
        # 注意：这里假设相机内参 f_x, f_y = 1
        ones = torch.ones_like(pred)
        pred_normal = torch.cat([-pred_grad_x, -pred_grad_y, ones], dim=1)
        gt_normal   = torch.cat([-gt_grad_x,   -gt_grad_y,   ones], dim=1)

        # 使用你成功的法线损失函数
        loss_normal = get_cycle_consistency_normal_loss(
            pred_normal, gt_normal, mask=mask_eroded.squeeze(1),
        )

    # --- 最终损失 ---
    # 加权求和
    # 权重可以根据你的任务进行调整，beta=1.0, gamma=0.0 是一个很好的起点
    total_loss = loss_silog + beta * loss_grad + gamma * loss_normal

    return total_loss


# ===================================================================
# 核心辅助函数：从深度图计算法线
# ===================================================================
# def depth_to_normals(depth, camera_intrinsics=None, eps=1e-8):
#     """
#     从深度图计算表面法线。
#     Args:
#         depth: [B, 1, H, W] 深度图
#         camera_intrinsics: [B, 3, 3] 相机内参矩阵 K。如果为 None，则在像素空间计算。
#     Returns:
#         normals: [B, 3, H, W] 法线图
#     """
#     B, _, H, W = depth.shape
#     device = depth.device
#     dtype = depth.dtype

#     # 创建像素坐标网格
#     y_coords, x_coords = torch.meshgrid(torch.arange(H, device=device, dtype=dtype),
#                                         torch.arange(W, device=device, dtype=dtype),
#                                         indexing='ij')
#     coords = torch.stack([x_coords, y_coords, torch.ones_like(x_coords)], dim=0).unsqueeze(0).repeat(B, 1, 1, 1)

#     if camera_intrinsics is not None:
#         # 反投影到相机坐标系
#         K_inv = torch.inverse(camera_intrinsics).view(B, 3, 3)
#         cam_coords = K_inv @ coords.to(torch.float32).view(B, 3, -1)
#         cam_coords = cam_coords.view(B, 3, H, W)
#         point_cloud = cam_coords * depth
#     else:
#         # 在像素空间计算（一个合理的近似）
#         point_cloud = torch.cat([coords[:, :2, ...], depth], dim=1)

#     # 使用 padding + 卷积计算梯度，更稳定
#     kernel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], device=device, dtype=dtype).view(1, 1, 3, 3) / 8.0
#     kernel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], device=device, dtype=dtype).view(1, 1, 3, 3) / 8.0
    
#     # 对点云的每个分量求梯度
#     point_cloud = point_cloud.to(torch.bfloat16)
#     p_x, p_y, p_z = point_cloud[:, 0:1], point_cloud[:, 1:2], point_cloud[:, 2:3]
    
#     tg_x = torch.cat([F.conv2d(p, kernel_x, padding='same') for p in (p_x, p_y, p_z)], dim=1)
#     tg_y = torch.cat([F.conv2d(p, kernel_y, padding='same') for p in (p_x, p_y, p_z)], dim=1)

#     # 计算法线 (cross product)
#     # tg_y x tg_x 的方向通常指向相机
#     normals = torch.cross(tg_y, tg_x, dim=1)

#     # 单位化
#     normals = normals / normals.norm(dim=1, keepdim=True).clamp_min(eps)
#     return normals


# # ===================================================================
# # 你的全新损失函数！
# # ===================================================================
# def get_cycle_consistency_depth_loss(pred_depth, gt_depth, mask=None, camera_intrinsics=None, depth_normalization="log",eps=1e-8,per_sample_weights=None):
#     """
#     将深度估计问题转化为法线几何一致性问题的损失函数。
#     1. 使用最小二乘法对齐 pred 和 gt_depth。
#     2. 将对齐后的 pred 和 gt_depth 转换为法线图。
#     3. 使用 get_cycle_consistency_normal_loss 计算法线间的损失。

#     Args:
#         pred: [B,3,H,W] or [B,H,W,3], 模型预测的深度图，值域 [-1, 1]
#         input_depth: [B,H,W] or [B,1,H,W], 原始的、未归一化的真实深度图
#         mask: None or [B,H,W], 有效像素区域
#         camera_intrinsics: 可选, [B, 3, 3] 相机内参, 用于更精确的法线计算
#         eps: 数值稳定项

#     Returns:
#         单标量 tensor (可反传)
#     """
#     # --- 形状、device、dtype 规范化 ---
#     device = pred_depth.device
#     dtype = pred_depth.dtype
#     if pred_depth.ndim ==4:
#         pred_depth = torch.mean(pred_depth, dim=1)  # [B, H, W]

#     if gt_depth.ndim == 4:
#         gt_depth = torch.mean(gt_depth, dim=1)  # [B, H, W]
        
#     B, H, W = pred_depth.shape
#     if depth_normalization == "log":
#         pred_depth = (pred_depth + 1.0) / 2.0 + eps
#         pred_depth = torch.exp(pred_depth)
#     elif depth_normalization == "sqrt":
#         pred_depth = (pred_depth + 1.0) / 2.0 + eps
#         pred_depth = pred_depth**2
#     elif depth_normalization == "disp":
#         pred_depth = (pred_depth + 1.0) / 2.0 + eps
#         pred_depth = 1.0/(pred_depth + eps)
#     elif depth_normalization == "sqrt_disp":
#         pred_depth = (pred_depth + 1.0) / 2.0 + eps
#         pred_depth = 1.0/(pred_depth**2 + eps)
        
#     # --- Mask 处理 ---
#     if mask is None:
#         mask = (gt_depth > eps).detach()
#     else:
#         # if mask.dim() == 3:
#             # mask = mask.unsqueeze(1)
#         mask = mask.bool().to(device)

#     # --- 核心步骤 1: 最小二乘法对齐尺度和偏移 ---
#     aligned_pred_depths = []
#     for i in range(B): # 逐个样本处理
#         p = pred_depth[i][mask[i]]
#         g = gt_depth[i][mask[i]]

#         if p.numel() < 2: # 有效点太少，无法拟合
#             aligned_pred_depths.append(pred_depth[i])
#             continue

#         # 构造最小二乘问题 y = s*x + t, A*[s,t]^T = y
#         A = torch.stack([p, torch.ones_like(p)], dim=1)
#         y = g

#         # 使用 torch.linalg.lstsq 求解，稳定且高效
#         try:
#             solution = torch.linalg.lstsq(A.to(torch.float32), y.to(torch.float32)).solution
#             s, t = solution[0], solution[1]
#         except torch.linalg.LinAlgError:
#             # 如果矩阵奇异，退化为只匹配均值
#             s = torch.tensor(1.0, device=device, dtype=dtype)
#             t = g.mean() - p.mean()

#         # 对齐预测值，不让梯度流过 s 和 t
#         aligned_pred_depths.append(pred_depth[i] * s.detach() + t.detach())

#     aligned_pred_depth = torch.stack(aligned_pred_depths)

#     # --- 核心步骤 2: 从对齐后的深度图计算法线 ---
#     # pred_normals = depth_to_normals(aligned_pred_depth, camera_intrinsics, k=5, d=1, gamma=0.05, min_nghbr=4, eps=eps)
#     # gt_normals = depth_to_normals(gt_depth, camera_intrinsics, k=5, d=1, gamma=0.05, min_nghbr=4, eps=eps)
#     # pred_normals = depth_to_normals(aligned_pred_depth, camera_intrinsics,)
#     # gt_normals = depth_to_normals(gt_depth, camera_intrinsics)
    
#     # # --- 核心步骤 3: 复用成功的法线损失 ---
#     # # 注意，法线计算在边缘处可能无效，mask需要结合深度mask
#     # final_mask = mask & (gt_depth > eps) & (pred_normals.norm(dim=1, keepdim=True) > eps) & (gt_normals.norm(dim=1, keepdim=True) > eps)
    
#     # loss = get_cycle_consistency_normal_loss(pred_normals, gt_normals, final_mask)
#     if per_sample_weights is None:
#         per_sample_weights = torch.ones((B,), device=device, dtype=dtype)
#     loss = per_sample_weights.view(B,1,1)*(aligned_pred_depth-gt_depth)**2*mask  # [B,H,W] -> [B]
#     return loss.mean().clip(1e-6,1.0)

import torch
import math

def get_cycle_consistency_normal_loss(pred, gt, mask=None, eps=1e-8, per_sample_weights=None):
    """
    稳定版的法线角度损失（可直接反传）
    Args:
        pred: [B,3,H,W] 或 [B,H,W,3]
        gt:   [B,3,H,W] 或 [B,H,W,3]
        mask: None 或 [B,H,W] / [B,1,H,W] / [B,H,W,1]
        eps:  数值稳定项
        per_sample_weights: None 或 [B]，每个样本的权重
    Returns:
        单标量 tensor（可反传）
    """
    # --- 形状、device、dtype 规范化 ---
    if pred.dim() == 4 and pred.shape[1] != 3 and pred.shape[-1] == 3:
        pred = pred.permute(0, 3, 1, 2)
    if gt.dim() == 4 and gt.shape[1] != 3 and gt.shape[-1] == 3:
        gt = gt.permute(0, 3, 1, 2)

    B, C, H, W = pred.shape
    assert C == 3, "输入必须是3通道法线向量"

    device = pred.device
    dtype = pred.dtype

    # --- 单位化（防止除零） ---
    pred = pred / pred.norm(dim=1, keepdim=True).clamp_min(eps)
    gt   = gt   / gt.norm(dim=1, keepdim=True).clamp_min(eps)

    # --- mask 处理 ---
    if mask is None:
        mask = torch.ones((B, H, W), dtype=torch.bool, device=device)
    if mask.dim() == 4:
        mask = mask.squeeze(1)
    if mask.dim() == 3 and mask.shape[1] == 1:  # 兼容 [B,1,H,W]
        mask = mask.squeeze(1)
    mask = mask.bool().to(device)

    # flatten为 [B,3,N] / [B,N]
    pred = pred.view(B, 3, -1)
    gt   = gt.view(B, 3, -1)
    mask = mask.view(B, -1).bool()  # [B, N]

    # --- 使用 dot 和 cross -> atan2 更稳定 ---
    dot = (pred * gt).sum(dim=1)  # [B, N], 取代 acos 的 cos 值
    cross = torch.cross(pred, gt, dim=1)  # [B,3,N]
    sin_val = cross.norm(dim=1)  # [B, N]

    # clamp 防止极端数值（但不把 dot 强行压到 ±1）
    dot = dot.clamp(-1.0 + 1e-7, 1.0 - 1e-7)
    sin_val = sin_val.clamp_min(1e-12)

    # angle in degrees
    angle = torch.atan2(sin_val, dot) * (180.0 / math.pi)  # [B, N]

    # --- 按 mask 计算 mean / accuracy（避免空mask导致 NaN） ---
    mask_f = mask.float()  # [B, N]
    counts_per_batch = mask_f.sum(dim=1)  # [B]
    # total = counts_per_batch.sum()  # scalar

    # if total.item() == 0:
    #     # 没有有效像素：返回 0 且保留梯度（避免图断裂）
    #     return torch.zeros((), device=device, dtype=dtype, requires_grad=True)
    # # mean angular error over masked elements
    # loss = (angle * mask_f ).sum() / total
    if per_sample_weights is None:
        per_sample_weights = torch.ones((B,), device=device, dtype=dtype)
    loss = (angle * mask_f).sum(dim=1) / counts_per_batch.clamp_min(1)  # [B]
    loss = (loss * per_sample_weights).mean()
    return loss / 100.0  # scale down for稳定（按你原代码保留）


def get_cycle_consistency_matting_loss(pred_alpha, gt_alpha, trimap):
    """
    一个集成的、可微的Matting损失函数 (PyTorch, Batch version)。
    
    此函数将SAD、MSE和Gradient Loss的计算合并在一起，并对未知区域进行操作。
    它被设计为简单、直接，易于集成到训练循环中。

    Args:
        pred_alpha (torch.Tensor): 预测的alpha matte, shape [B, 1, H, W], 值在 [0, 1] 之间。
        gt_alpha (torch.Tensor): 真实的alpha matte, shape [B, 1, H, W], 值在 [0, 1] 之间。
        trimap (torch.Tensor): Trimap, shape [B, 1, H, W], 值为 0, 128, 255。

    Returns:
        torch.Tensor: 一个用于反向传播的标量损失值。
    """
    # --- 1. 预处理: 创建未知区域的掩码 ---
    # 未知区域是trimap中值为128的像素
    if pred_alpha.ndim == 4:
        pred_alpha = torch.mean(pred_alpha, dim=1, keepdim=True)  # [B,1,H,W]
    if gt_alpha.ndim == 4:
        gt_alpha = torch.mean(gt_alpha, dim=1, keepdim=True)      # [B,1,H,W]
    if trimap.ndim == 3:
        trimap = trimap.unsqueeze(1)  # [B,1,H,W]
    unknown_mask = ((trimap >= 127.5) & (trimap <= 128.5)).float()

    # --- 2. SAD Loss (Sum of Absolute Differences) ---
    error_map_sad = torch.abs(pred_alpha - gt_alpha)
    # 对每个样本的未知区域求和，并按原始实现进行缩放
    loss_sad_per_sample = (error_map_sad * unknown_mask).sum(dim=(1, 2, 3)) / 1000.0

    # --- 3. MSE Loss (Mean Squared Error) ---
    error_map_mse = (pred_alpha - gt_alpha) ** 2
    # 对每个样本的未知区域求和，然后除以该区域的像素数
    loss_mse_per_sample = (error_map_mse * unknown_mask).sum(dim=(1, 2, 3)) / (unknown_mask.sum(dim=(1, 2, 3)) + 1e-8)

    # --- 4. Gradient Loss ---
    # 在函数内部动态创建高斯梯度核，以匹配输入张量的设备
    device = pred_alpha.device
    sigma = 1.4
    
    # 创建高斯导数核
    epsilon = 1e-2
    halfsize = math.ceil(sigma * math.sqrt(-2 * math.log(math.sqrt(2 * math.pi) * sigma * epsilon)))
    size = 2 * halfsize + 1
    coords = torch.arange(-halfsize, halfsize + 1, dtype=torch.float32, device=device)
    
    gauss_vals = torch.exp(-coords**2 / (2 * sigma**2)) / (sigma * math.sqrt(2 * math.pi))
    dgauss_vals = -coords * gauss_vals / (sigma**2)
    
    # x和y方向的核
    hx = (gauss_vals.unsqueeze(1) * dgauss_vals.unsqueeze(0))
    hy = hx.t()
    
    # 归一化
    hx = hx / torch.sqrt(torch.sum(torch.abs(hx) * torch.abs(hx)))
    hy = hy / torch.sqrt(torch.sum(torch.abs(hy) * torch.abs(hy)))
    
    # 为conv2d调整形状: [out_channels, in_channels, H, W]
    kernel_x = hx.to(pred_alpha.dtype).unsqueeze(0).unsqueeze(0)
    kernel_y = hy.to(pred_alpha.dtype).unsqueeze(0).unsqueeze(0)

    # 使用卷积计算梯度
    pred_gx = F.conv2d(pred_alpha, kernel_x, padding='same')
    pred_gy = F.conv2d(pred_alpha, kernel_y, padding='same')
    gt_gx = F.conv2d(gt_alpha, kernel_x, padding='same')
    gt_gy = F.conv2d(gt_alpha, kernel_y, padding='same')

    # 计算梯度幅值
    pred_amp = torch.sqrt(pred_gx**2 + pred_gy**2)
    gt_amp = torch.sqrt(gt_gx**2 + gt_gy**2)

    # 计算梯度损失
    error_map_grad = (pred_amp - gt_amp) ** 2
    loss_grad_per_sample = (error_map_grad * unknown_mask).sum(dim=(1, 2, 3)) / 1000.0

    # --- 5. 组合损失并在Batch维度上取平均 ---
    # 你可以根据需要为各项损失添加权重, e.g., 1.0 * loss_sad + 0.5 * loss_mse + ...
    total_loss_per_sample = loss_sad_per_sample + loss_mse_per_sample + loss_grad_per_sample
    final_loss = total_loss_per_sample.mean()

    return final_loss

def get_disperse_loss(eta, tau=1.0):
    """
    计算 Dispersive Loss.
    公式: L_disp = log E_{i,j} [exp(-|| η_i - η_j ||_2^2 / τ)]
    
    参数:
        eta (torch.Tensor): 一个批次的中间特征，形状为 (batch_size, seq_len, dim) 或 (batch_size, features)。
        tau (float): 温度超参数，根据论文设置为 1.0。
    
    返回:
        torch.Tensor: 计算出的 dispersive loss (标量).
    """
    batch_size = eta.shape[0]
    # 如果批次大小小于等于1，无法计算配对距离，损失为0
    if batch_size <= 1:
        return torch.tensor(0.0, device=eta.device, dtype=eta.dtype)

    # 如果特征是序列化的 (B, N, D)，则将其展平为 (B, N*D)
    if eta.dim() > 2:
        eta = eta.view(batch_size, -1)

    # 高效计算批次内样本间的成对L2距离的平方
    # (x-y)^2 = x^2 - 2xy + y^2
    eta_norm_sq = torch.sum(eta**2, dim=1, keepdim=True)
    # dist_sq 是一个 (batch_size, batch_size) 的矩阵，存储了距离的平方
    dist_sq = eta_norm_sq - 2 * torch.mm(eta, eta.t()) + eta_norm_sq.t()
    
    # 确保数值稳定性，距离的平方应为非负数
    dist_sq = torch.clamp(dist_sq, min=0.0)

    # 计算 e^(-d^2/τ)
    loss_matrix = torch.exp(-dist_sq / tau)
    
    # 计算期望 E_{i,j}，即对所有 B*B 配对取均值
    expectation = torch.mean(loss_matrix)
    
    # 最终损失是期望的对数
    loss = torch.log(expectation)
    
    return loss