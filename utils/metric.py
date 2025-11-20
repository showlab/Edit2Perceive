
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

# Adapted from: https://github.com/victoresque/pytorch-template/blob/master/utils/util.py
class MetricTracker:
    def __init__(self, *keys, writer=None):
        self.writer = writer
        self._data = pd.DataFrame(index=keys, columns=["total", "counts", "average"])
        self.reset()

    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key, value, n=1):
        if self.writer is not None:
            self.writer.add_scalar(key, value)
        self._data.loc[key, "total"] += value * n
        self._data.loc[key, "counts"] += n
        self._data.loc[key, "average"] = self._data.total[key] / self._data.counts[key]

    def avg(self, key):
        return self._data.average[key]

    def result(self):
        return dict(self._data.average)


# -------------------- Depth Metrics --------------------


def abs_relative_difference(output, target, valid_mask=None):
    actual_output = output
    actual_target = target
    abs_relative_diff = torch.abs(actual_output - actual_target) / actual_target
    if valid_mask is not None:
        abs_relative_diff[~valid_mask] = 0
        n = valid_mask.sum((-1, -2))
    else:
        n = output.shape[-1] * output.shape[-2]
    abs_relative_diff = torch.sum(abs_relative_diff, (-1, -2)) / n
    return abs_relative_diff.mean()


def squared_relative_difference(output, target, valid_mask=None):
    actual_output = output
    actual_target = target
    square_relative_diff = (
        torch.pow(torch.abs(actual_output - actual_target), 2) / actual_target
    )
    if valid_mask is not None:
        square_relative_diff[~valid_mask] = 0
        n = valid_mask.sum((-1, -2))
    else:
        n = output.shape[-1] * output.shape[-2]
    square_relative_diff = torch.sum(square_relative_diff, (-1, -2)) / n
    return square_relative_diff.mean()


def rmse_linear(output, target, valid_mask=None):
    actual_output = output
    actual_target = target
    diff = actual_output - actual_target
    if valid_mask is not None:
        diff[~valid_mask] = 0
        n = valid_mask.sum((-1, -2))
    else:
        n = output.shape[-1] * output.shape[-2]
    diff2 = torch.pow(diff, 2)
    mse = torch.sum(diff2, (-1, -2)) / n
    rmse = torch.sqrt(mse)
    return rmse.mean()


def rmse_log(output, target, valid_mask=None):
    diff = torch.log(output) - torch.log(target)
    if valid_mask is not None:
        diff[~valid_mask] = 0
        n = valid_mask.sum((-1, -2))
    else:
        n = output.shape[-1] * output.shape[-2]
    diff2 = torch.pow(diff, 2)
    mse = torch.sum(diff2, (-1, -2)) / n  # [B]
    rmse = torch.sqrt(mse)
    return rmse.mean()


def log10(output, target, valid_mask=None):
    if valid_mask is not None:
        diff = torch.abs(
            torch.log10(output[valid_mask]) - torch.log10(target[valid_mask])
        )
    else:
        diff = torch.abs(torch.log10(output) - torch.log10(target))
    return diff.mean()


# Adapted from: https://github.com/imran3180/depth-map-prediction/blob/master/main.py
def threshold_percentage(output, target, threshold_val, valid_mask=None):
    d1 = output / target
    d2 = target / output
    max_d1_d2 = torch.max(d1, d2)
    zero = torch.zeros(*output.shape)
    one = torch.ones(*output.shape)
    bit_mat = torch.where(max_d1_d2.cpu() < threshold_val, one, zero)
    if valid_mask is not None:
        bit_mat[~valid_mask] = 0
        n = valid_mask.sum((-1, -2))
    else:
        n = output.shape[-1] * output.shape[-2]
    count_mat = torch.sum(bit_mat, (-1, -2))
    threshold_mat = count_mat / n.cpu()
    return threshold_mat.mean()


def delta1_acc(pred, gt, valid_mask):
    return threshold_percentage(pred, gt, 1.25, valid_mask)


def delta2_acc(pred, gt, valid_mask):
    return threshold_percentage(pred, gt, 1.25**2, valid_mask)


def delta3_acc(pred, gt, valid_mask):
    return threshold_percentage(pred, gt, 1.25**3, valid_mask)


def i_rmse(output, target, valid_mask=None):
    output_inv = 1.0 / output
    target_inv = 1.0 / target
    diff = output_inv - target_inv
    if valid_mask is not None:
        diff[~valid_mask] = 0
        n = valid_mask.sum((-1, -2))
    else:
        n = output.shape[-1] * output.shape[-2]
    diff2 = torch.pow(diff, 2)
    mse = torch.sum(diff2, (-1, -2)) / n  # [B]
    rmse = torch.sqrt(mse)
    return rmse.mean()


def silog_rmse(depth_pred, depth_gt, valid_mask=None):
    diff = torch.log(depth_pred) - torch.log(depth_gt)
    if valid_mask is not None:
        diff[~valid_mask] = 0
        n = valid_mask.sum((-1, -2))
    else:
        n = depth_gt.shape[-2] * depth_gt.shape[-1]

    diff2 = torch.pow(diff, 2)

    first_term = torch.sum(diff2, (-1, -2)) / n
    second_term = torch.pow(torch.sum(diff, (-1, -2)), 2) / (n**2)
    loss = torch.sqrt(torch.mean(first_term - second_term)) * 100
    return loss


# -------------------- Normals Metrics --------------------
def get_padding(orig_H, orig_W):
    """ returns how the input of shape (orig_H, orig_W) should be padded
        this ensures that both H and W are divisible by 32
    """
    if orig_W % 32 == 0:
        l = 0
        r = 0
    else:
        new_W = 32 * ((orig_W // 32) + 1)
        l = (new_W - orig_W) // 2
        r = (new_W - orig_W) - l

    if orig_H % 32 == 0:
        t = 0
        b = 0
    else:
        new_H = 32 * ((orig_H // 32) + 1)
        t = (new_H - orig_H) // 2
        b = (new_H - orig_H) - t
    return l, r, t, b

def pad_input(img, intrins, lrtb=(0,0,0,0)):
    """ pad input image
        img should be a torch tensor of shape (B, 3, H, W)
        intrins should be a torch tensor of shape (B, 3, 3)
    """
    l, r, t, b = lrtb
    if l+r+t+b != 0:
        pad_value_R = (0 - 0.485) / 0.229
        pad_value_G = (0 - 0.456) / 0.224
        pad_value_B = (0 - 0.406) / 0.225

        img_R = F.pad(img[:,0:1,:,:], (l, r, t, b), mode="constant", value=pad_value_R)
        img_G = F.pad(img[:,1:2,:,:], (l, r, t, b), mode="constant", value=pad_value_G)
        img_B = F.pad(img[:,2:3,:,:], (l, r, t, b), mode="constant", value=pad_value_B)

        img = torch.cat([img_R, img_G, img_B], dim=1)

        if intrins is not None:
            intrins[:, 0, 2] += l
            intrins[:, 1, 2] += t
    return img, intrins

def compute_cosine_error(pred_norm, gt_norm, masked=True,eps=1e-3):
    # allow for numpy array input
    if isinstance(pred_norm, np.ndarray):
        pred_norm = torch.from_numpy(pred_norm.transpose(2, 0, 1))  # [H,W,3] -> [3,H,W]
    if isinstance(gt_norm, np.ndarray):
        gt_norm = torch.from_numpy(gt_norm.transpose(2, 0, 1))  # [H,W,3] -> [3,H,W]
    else:
        if len(pred_norm.shape) == 4:
            pred_norm = pred_norm.squeeze(0)
        if len(gt_norm.shape) == 4:
            gt_norm = gt_norm.squeeze(0)

    # shape must be [3,H,W]
    assert (gt_norm.shape[0] == 3) and (
        pred_norm.shape[0] == 3
    ), "Channel dim should be the first dimension!"
    # mask out the zero vectors, otherwise torch.cosine_similarity computes 90° as error
    if masked:
        ch, h, w = gt_norm.shape
        
        mask = torch.norm(gt_norm.float(), dim=0) > 1e-3
        if gt_norm.min()>-eps and gt_norm.max()<254+eps:
            gt_norm = gt_norm / 254.0*2.0 - 1.0
        if gt_norm.min()>-eps and gt_norm.max()<255+eps:
            gt_norm = gt_norm / 255.0*2.0 - 1.0
        if pred_norm.min()>-eps and pred_norm.max()<1+eps:
            pred_norm = (pred_norm - 0.5) * 2
        # print(f"shape: {pred_norm.shape}, {gt_norm.shape}, pred range:{pred_norm.min()}~{pred_norm.max()}, gt range:{gt_norm.min()}~{gt_norm.max()}")
        pred_norm = pred_norm[:, mask.view(h, w)]
        gt_norm = gt_norm[:, mask.view(h, w)]
    pred_error = torch.cosine_similarity(pred_norm, gt_norm, dim=0)
    pred_error = torch.clamp(pred_error, min=-1.0, max=1.0)
    pred_error = torch.acos(pred_error) * 180.0 / np.pi  # (H, W)

    return (
        pred_error.view(-1).detach().cpu().numpy()
    )  # flatten so can directly input to compute_normal_metrics()


def mean_angular_error(cosine_error):
    return round(np.average(cosine_error), 4)


def median_angular_error(cosine_error):
    return round(np.median(cosine_error), 4)


def rmse_angular_error(cosine_error):
    num_pixels = cosine_error.shape[0]
    return round(np.sqrt(np.sum(cosine_error * cosine_error) / num_pixels), 4)


def sub5_error(cosine_error):
    num_pixels = cosine_error.shape[0]
    return round(100.0 * (np.sum(cosine_error < 5) / num_pixels), 4)


def sub7_5_error(cosine_error):
    num_pixels = cosine_error.shape[0]
    return round(100.0 * (np.sum(cosine_error < 7.5) / num_pixels), 4)


def sub11_25_error(cosine_error):
    num_pixels = cosine_error.shape[0]
    return round(100.0 * (np.sum(cosine_error < 11.25) / num_pixels), 4)


def sub22_5_error(cosine_error):
    num_pixels = cosine_error.shape[0]
    return round(100.0 * (np.sum(cosine_error < 22.5) / num_pixels), 4)


def sub30_error(cosine_error):
    num_pixels = cosine_error.shape[0]
    return round(100.0 * (np.sum(cosine_error < 30) / num_pixels), 4)

def compute_normal_metrics(pred_norm, gt_norm, masked=True):
    """
    Computes the normal metrics for the predicted and ground truth normals.
    :param pred_norm: Predicted normals, shape [3,H,W]
    :param gt_norm: Ground truth normals, shape [3,H,W]
    :param masked: If True, mask out zero vectors in the normals
    :return: Dictionary with computed metrics
    """
    cosine_error = compute_cosine_error(pred_norm, gt_norm, masked)
    
    return [mean_angular_error(cosine_error),
            median_angular_error(cosine_error),
            rmse_angular_error(cosine_error),
            sub5_error(cosine_error),
            sub7_5_error(cosine_error),
            sub11_25_error(cosine_error),
            sub22_5_error(cosine_error),
            sub30_error(cosine_error)]

# -------------------- IID Metrics --------------------


def compute_iid_metric(pred, gt, target_name, metric_name, metric, valid_mask=None):
    # Shading and residual are up-to-scale. We first scale-align them to the gt
    # and map them to the range [0,1] for metric computation
    if target_name == "shading" or target_name == "residual":
        alignment_scale = compute_alignment_scale(pred, gt, valid_mask)
        pred = alignment_scale * pred
        # map to [0,1]
        pred, gt = quantile_map(pred, gt, valid_mask)

    if len(pred.shape) == 3:
        pred = pred.unsqueeze(0)
    if len(gt.shape) == 3:
        gt = gt.unsqueeze(0)
    if valid_mask is not None:
        if len(valid_mask.shape) == 3:
            valid_mask = valid_mask.unsqueeze(0)
        if metric_name == "psnr":
            return metric(pred[valid_mask], gt[valid_mask]).item()
        # for SSIM and LPIPs set the invalid pixels to zero
        else:
            invalid_mask = ~valid_mask
            pred[invalid_mask] = 0
            gt[invalid_mask] = 0

    return metric(pred, gt).item()


# compute least-squares alignment scale to align shading/residual prediction to gt
def compute_alignment_scale(pred, gt, valid_mask=None):
    pred = pred.squeeze()
    gt = gt.squeeze()
    assert pred.shape[0] == 3 and gt.shape[0] == 3, "First dim should be channel dim"

    if valid_mask is not None:
        valid_mask = valid_mask.squeeze()
        pred = pred[valid_mask]
        gt = gt[valid_mask]

    A_flattened = pred.view(-1, 1)
    b_flattened = gt.view(-1, 1)
    # Solve the least squares problem
    x, residuals, rank, s = torch.linalg.lstsq(A_flattened.float(), b_flattened.float())
    return x


def quantile_map(pred, gt, valid_mask=None):
    pred = pred.squeeze()
    gt = gt.squeeze()
    assert gt.shape[0] == 3, "channel dim must be first dim"

    percentile = 90
    brightness_nth_percentile_desired = 0.8
    brightness = 0.3 * gt[0, :, :] + 0.59 * gt[1, :, :] + 0.11 * gt[2, :, :]

    if valid_mask is not None:
        valid_mask = valid_mask.squeeze()
        brightness = brightness[valid_mask[0]]
    else:
        brightness = brightness.flatten()

    eps = 0.0001

    brightness_nth_percentile_current = torch.quantile(brightness, percentile / 100.0)

    if brightness_nth_percentile_current < eps:
        scale = 0
    else:
        scale = float(
            brightness_nth_percentile_desired / brightness_nth_percentile_current
        )

    # Apply scaling to ground truth and prediction
    gt_mapped = torch.clamp(scale * gt, 0, 1).unsqueeze(0)  # [1,3,H,W]
    pred_mapped = torch.clamp(scale * pred, 0, 1).unsqueeze(0)  # [1,3,H,W]

    return pred_mapped, gt_mapped


# # Matting Metrics
# """
# Reimplement evaluation.mat provided by Adobe in python
# Output of `compute_gradient_loss` is sightly different from the MATLAB version provided by Adobe (less than 0.1%)
# Output of `compute_connectivity_error` is smaller than the MATLAB version (~5%, maybe MATLAB has a different algorithm)
# So do not report results calculated by these functions in your paper.
# Evaluate your inference with the MATLAB file `DIM_evaluation_code/evaluate.m`.

# by Yaoyi Li
# """

# from scipy.ndimage import convolve
# import numpy as np
# from skimage.measure import label


# def gauss(x, sigma):
#     y = np.exp(-x ** 2 / (2 * sigma ** 2)) / (sigma * np.sqrt(2 * np.pi))
#     return y


# def dgauss(x, sigma):
#     y = -x * gauss(x, sigma) / (sigma ** 2)
#     return y


# def gaussgradient(im, sigma):
#     epsilon = 1e-2
#     halfsize = np.ceil(sigma * np.sqrt(-2 * np.log(np.sqrt(2 * np.pi) * sigma * epsilon))).astype(np.int32)
#     size = 2 * halfsize + 1
#     hx = np.zeros((size, size))
#     for i in range(0, size):
#         for j in range(0, size):
#             u = [i - halfsize, j - halfsize]
#             hx[i, j] = gauss(u[0], sigma) * dgauss(u[1], sigma)

#     hx = hx / np.sqrt(np.sum(np.abs(hx) * np.abs(hx)))
#     hy = hx.transpose()

#     gx = convolve(im, hx, mode='nearest')
#     gy = convolve(im, hy, mode='nearest')

#     return gx, gy


# def compute_gradient_loss(pred, target, trimap):

#     pred_x, pred_y = gaussgradient(pred, 1.4)
#     target_x, target_y = gaussgradient(target, 1.4)

#     pred_amp = np.sqrt(pred_x ** 2 + pred_y ** 2)
#     target_amp = np.sqrt(target_x ** 2 + target_y ** 2)

#     error_map = (pred_amp - target_amp) ** 2
#     loss = np.sum(error_map[trimap == 128])

#     return loss / 1000.


# def getLargestCC(segmentation):
#     labels = label(segmentation, connectivity=1)
#     largestCC = labels == np.argmax(np.bincount(labels.flat))
#     return largestCC


# def compute_connectivity_error(pred, target, trimap, step):
#     h, w = pred.shape

#     thresh_steps = list(np.arange(0, 1 + step, step))
#     l_map = np.ones_like(pred, dtype=np.float32) * -1
#     for i in range(1, len(thresh_steps)):
#         pred_alpha_thresh = (pred >= thresh_steps[i]).astype(np.uint8)
#         target_alpha_thresh = (target >= thresh_steps[i]).astype(np.uint8)

#         omega = getLargestCC(pred_alpha_thresh * target_alpha_thresh).astype(np.uint8)
#         flag = ((l_map == -1) & (omega == 0)).astype(np.uint8)
#         l_map[flag == 1] = thresh_steps[i - 1]

#     l_map[l_map == -1] = 1

#     pred_d = pred - l_map
#     target_d = target - l_map
#     pred_phi = 1 - pred_d * (pred_d >= 0.15).astype(np.uint8)
#     target_phi = 1 - target_d * (target_d >= 0.15).astype(np.uint8)
#     loss = np.sum(np.abs(pred_phi - target_phi)[trimap == 128])

#     return loss / 1000.


# def compute_mse_loss(pred, target, trimap):
#     error_map = (pred - target)
#     loss = np.sum((error_map ** 2) * (trimap == 128)) / (np.sum(trimap == 128) + 1e-8)

#     return loss


# def comput_sad_loss(pred, target, trimap):
#     error_map = np.abs(pred - target)
#     loss = np.sum(error_map * (trimap == 128))

#     return loss / 1000, np.sum(trimap == 128) / 1000

# def compute_matting_metrics(pred, gt, trimap):
#     """
#     Computes the matting metrics for the predicted and ground truth alpha mattes.
#     :param pred: Predicted alpha matte, shape [H,W]
#     :param gt: Ground truth alpha matte, shape [H,W]
#     :param trimap: Trimap, shape [H,W]
#     :return: Dictionary with computed metrics
#     """
#     if isinstance(pred, torch.Tensor):
#         pred = pred.squeeze().cpu().numpy()
#     if isinstance(gt, torch.Tensor):
#         gt = gt.squeeze().cpu().numpy()
#     if isinstance(trimap, torch.Tensor):
#         trimap = trimap.squeeze().cpu.numpy()

#     sad_loss, valid_pixel_count = comput_sad_loss(pred, gt, trimap)
#     mse_loss = compute_mse_loss(pred, gt, trimap)
#     gradient_loss = compute_gradient_loss(pred, gt, trimap)
#     connectivity_error = compute_connectivity_error(pred, gt, trimap, 0.1)

#     return [sad_loss,mse_loss,gradient_loss, connectivity_error]


"""
Rethinking Portrait Matting with Privacy Preserving

Copyright (c) 2022, Sihan Ma (sima7436@uni.sydney.edu.au) and Jizhizi Li (jili8515@uni.sydney.edu.au)
Licensed under the MIT License (see LICENSE for details)
Github repo: https://github.com/ViTAE-Transformer/P3M-Net
Paper link: https://arxiv.org/abs/2203.16828

"""

import numpy as np
from skimage.measure import label
import scipy.ndimage.morphology
##############################
### Test loss for matting
##############################

def calculate_sad_mse_mad(predict_old,alpha,trimap):
    predict = np.copy(predict_old)
    pixel = float((trimap == 128).sum())
    predict[trimap == 255] = 1.
    predict[trimap == 0  ] = 0.
    sad_diff = np.sum(np.abs(predict - alpha))/1000
    if pixel==0:
        pixel = trimap.shape[0]*trimap.shape[1]-float((trimap==255).sum())-float((trimap==0).sum())
    mse_diff = np.sum((predict - alpha) ** 2)/pixel
    mad_diff = np.sum(np.abs(predict - alpha))/pixel
    return sad_diff, mse_diff, mad_diff
    
def calculate_sad_mse_mad_whole_img(predict, alpha):
    pixel = predict.shape[0]*predict.shape[1]
    sad_diff = np.sum(np.abs(predict - alpha))/1000
    mse_diff = np.sum((predict - alpha) ** 2)/pixel
    mad_diff = np.sum(np.abs(predict - alpha))/pixel
    return sad_diff, mse_diff, mad_diff	

def calculate_sad_fgbg(predict, alpha, trimap):
    sad_diff = np.abs(predict-alpha)
    weight_fg = np.zeros(predict.shape)
    weight_bg = np.zeros(predict.shape)
    weight_trimap = np.zeros(predict.shape)
    weight_fg[trimap==255] = 1.
    weight_bg[trimap==0  ] = 1.
    weight_trimap[trimap==128  ] = 1.
    sad_fg = np.sum(sad_diff*weight_fg)/1000
    sad_bg = np.sum(sad_diff*weight_bg)/1000
    sad_trimap = np.sum(sad_diff*weight_trimap)/1000
    return sad_fg, sad_bg

# def compute_gradient_whole_image(pd, gt):
#     from scipy.ndimage import gaussian_filter
#     pd_x = gaussian_filter(pd, sigma=1.4, order=[1, 0], output=np.float32)
#     pd_y = gaussian_filter(pd, sigma=1.4, order=[0, 1], output=np.float32)
#     gt_x = gaussian_filter(gt, sigma=1.4, order=[1, 0], output=np.float32)
#     gt_y = gaussian_filter(gt, sigma=1.4, order=[0, 1], output=np.float32)
#     pd_mag = np.sqrt(pd_x**2 + pd_y**2)
#     gt_mag = np.sqrt(gt_x**2 + gt_y**2)

#     error_map = np.square(pd_mag - gt_mag)
#     loss = np.sum(error_map) / 10
#     return loss

# def compute_connectivity_loss_whole_image(pd, gt, trimap=None, step=0.1):
#     from scipy.ndimage import morphology
#     from skimage.measure import label, regionprops
#     h, w = pd.shape
#     thresh_steps = np.arange(0, 1.1, step)
#     l_map = -1 * np.ones((h, w), dtype=np.float32)
#     lambda_map = np.ones((h, w), dtype=np.float32)
#     for i in range(1, thresh_steps.size):
#         pd_th = pd >= thresh_steps[i]
#         gt_th = gt >= thresh_steps[i]
#         label_image = label(pd_th & gt_th, connectivity=1)
#         cc = regionprops(label_image)
#         size_vec = np.array([c.area for c in cc])
#         if len(size_vec) == 0:
#             continue
#         max_id = np.argmax(size_vec)
#         coords = cc[max_id].coords
#         omega = np.zeros((h, w), dtype=np.float32)
#         omega[coords[:, 0], coords[:, 1]] = 1
#         flag = (l_map == -1) & (omega == 0)
#         l_map[flag == 1] = thresh_steps[i-1]
#         dist_maps = morphology.distance_transform_edt(omega==0)
#         dist_maps = dist_maps / dist_maps.max()
#     l_map[l_map == -1] = 1
#     d_pd = pd - l_map
#     d_gt = gt - l_map
#     phi_pd = 1 - d_pd * (d_pd >= 0.15).astype(np.float32)
#     phi_gt = 1 - d_gt * (d_gt >= 0.15).astype(np.float32)
#     if trimap is not None:
#         loss = np.sum(np.abs(phi_pd - phi_gt) * (trimap == 128)) / 1000
#     else:
#         loss = np.sum(np.abs(phi_pd - phi_gt)) / 1000
#     return loss


def gauss(x, sigma):
    y = np.exp(-x ** 2 / (2 * sigma ** 2)) / (sigma * np.sqrt(2 * np.pi))
    return y


def dgauss(x, sigma):
    y = -x * gauss(x, sigma) / (sigma ** 2)
    return y


def gaussgradient(im, sigma):
    epsilon = 1e-2
    halfsize = np.ceil(sigma * np.sqrt(-2 * np.log(np.sqrt(2 * np.pi) * sigma * epsilon))).astype(np.int_)
    size = 2 * halfsize + 1
    hx = np.zeros((size, size))
    for i in range(0, size):
        for j in range(0, size):
            u = [i - halfsize, j - halfsize]
            hx[i, j] = gauss(u[0], sigma) * dgauss(u[1], sigma)

    hx = hx / np.sqrt(np.sum(np.abs(hx) * np.abs(hx)))
    hy = hx.transpose()

    gx = scipy.ndimage.convolve(im, hx, mode='nearest')
    gy = scipy.ndimage.convolve(im, hy, mode='nearest')

    return gx, gy

def compute_gradient_loss(pred, target, trimap=None):

    if pred.dtype == np.uint8:
        pred = pred / 255.0
    if target.dtype == np.uint8:
        target = target / 255.0

    pred_x, pred_y = gaussgradient(pred, 1.4)
    target_x, target_y = gaussgradient(target, 1.4)

    pred_amp = np.sqrt(pred_x ** 2 + pred_y ** 2)
    target_amp = np.sqrt(target_x ** 2 + target_y ** 2)

    error_map = (pred_amp - target_amp) ** 2
    if trimap is not None:
        loss = np.sum(error_map[trimap == 128])
    else:
        loss = np.sum(error_map)
    return loss / 1000.


def getLargestCC(segmentation):
    labels = label(segmentation, connectivity=1)
    largestCC = labels == np.argmax(np.bincount(labels.flat))
    return largestCC


def compute_connectivity_error(pred, target, trimap=None, step=0.1):
    if pred.dtype == np.uint8:
        pred = pred / 255.0
    if target.dtype == np.uint8:
        target = target / 255.0
    h, w = pred.shape

    thresh_steps = list(np.arange(0, 1 + step, step))
    l_map = np.ones_like(pred, dtype=np.float32) * -1
    for i in range(1, len(thresh_steps)):
        pred_alpha_thresh = (pred >= thresh_steps[i]).astype(np.int_)
        target_alpha_thresh = (target >= thresh_steps[i]).astype(np.int_)

        omega = getLargestCC(pred_alpha_thresh * target_alpha_thresh).astype(np.int_)
        flag = ((l_map == -1) & (omega == 0)).astype(np.int_)
        l_map[flag == 1] = thresh_steps[i - 1]

    l_map[l_map == -1] = 1

    pred_d = pred - l_map
    target_d = target - l_map
    pred_phi = 1 - pred_d * (pred_d >= 0.15).astype(np.int_)
    target_phi = 1 - target_d * (target_d >= 0.15).astype(np.int_)
    if trimap is not None:
        loss = np.sum(np.abs(pred_phi - target_phi)[trimap == 128])
    else:
        loss = np.sum(np.abs(pred_phi - target_phi)) 

    return loss / 1000.

def compute_matting_metrics(pred, alpha, trimap=None,whole=False):
    if isinstance(pred, torch.Tensor):
        pred = pred.squeeze().cpu().numpy()
    if isinstance(alpha, torch.Tensor):
        alpha = alpha.squeeze().cpu().numpy()
    if isinstance(trimap, torch.Tensor):
        trimap = trimap.squeeze().cpu().numpy()
    sad, mse, mad = calculate_sad_mse_mad(pred, alpha, trimap)
    conn = compute_gradient_loss(pred, alpha, trimap)
    if whole:
        sad_whole, mse_whole, mad_whole = calculate_sad_mse_mad_whole_img(pred, alpha)
        sad_fg, sad_bg = calculate_sad_fgbg(pred, alpha, trimap)
        gradient_loss = compute_gradient_loss(pred, alpha)
        connectivity_loss = compute_connectivity_error(pred, alpha)

        # return [sad,sad_fg,sad_bg,mse,mad,sad_whole,mse_whole,mad_whole,gradient_loss,connectivity_loss]
        return [mse_whole,mad_whole,sad_whole,gradient_loss,connectivity_loss]
    return [sad,mse,mad,conn]