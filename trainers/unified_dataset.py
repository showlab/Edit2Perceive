import torch, torchvision, os, json, pandas
# import imageio
# import imageio.v3 as iio
from PIL import Image
import numpy as np
import cv2
import torch.nn.functional as F
import random
from typing import Union
from torchvision.transforms import functional as TF

import scipy.ndimage
from scipy.ndimage import label

# --- 预计算腐蚀/膨胀的核，避免在函数内重复生成，提高效率 ---
# 这对应于 GenMask 和 GenTrimap 中的 self.erosion_kernels
# 最大核尺寸可以根据需要调整
MAX_KERNEL_SIZE = 30
EROSION_KERNELS = [None] + [cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size)) for size in range(1, MAX_KERNEL_SIZE)]
# fix the all random seed for reproducibility
random.seed(42)
np.random.seed(42)


def gen_trimap(alpha: np.ndarray) -> np.ndarray:
    """
    根据 alpha 通道图生成 trimap.
    前景为255, 背景为0, 未知区域为128.

    Args:
        alpha (np.ndarray): 输入的单通道 alpha 图像 (0-255, uint8).

    Returns:
        np.ndarray: 生成的 trimap 图像 (uint8).
    """
    assert alpha.dtype == np.uint8, "Alpha channel must be of type uint8"
        
    k_size = random.choice(range(5, 15)) # 稍微增大随机核范围，效果更明显
    iterations = np.random.randint(5, 15) # 同样增大迭代次数
    
    # 从预计算的列表中获取核
    kernel = EROSION_KERNELS[k_size]
    
    dilated = cv2.dilate(alpha, kernel, iterations=iterations)
    eroded = cv2.erode(alpha, kernel, iterations=iterations)
    
    trimap = np.zeros(alpha.shape, dtype=np.uint8)
    trimap.fill(128)  # 未知区域
    trimap[eroded >= 254] = 255  # 确定前景 (使用254以防万一)
    trimap[dilated <= 1] = 0      # 确定背景 (使用1以防万一)
    # 归一化坐标
    coords = np.argwhere(trimap == 128)
    h, w = alpha.shape
    if coords.size == 0:
        # 如果没有未知区域，返回一个默认的小框
        coords = np.array([[0, 0], [1, 1]])
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    
    # 归一化坐标
    x_min, x_max = x_min / w, x_max / w
    y_min, y_max = y_min / h, y_max / h
    coords = np.array([x_min, y_min, x_max, y_max])
    return trimap, coords.astype(np.float32)


def gen_mask(alpha: np.ndarray, min_kernel_size: int = 15, max_kernel_size: int = 29) -> dict:
    """
    从 alpha 通道随机生成一个二值分割掩码 (mask).
    该过程包括阈值化、随机的腐蚀或膨胀操作.

    Args:
        alpha (np.ndarray): 输入的单通道 alpha 图像 (0.0-1.0, float).
        min_kernel_size (int): 用于腐蚀/膨胀的最小核尺寸.
        max_kernel_size (int): 用于腐蚀/膨胀的最大核尺寸.

    Returns:
        dict: 一个包含 'mask' 和 'mask_coords' 的字典.
              'mask': 生成的二值掩码 (h, w), float32.
              'mask_coords': 归一化的掩码边界框 [x_min, y_min, x_max, y_max], np.ndarray.
    """
    # 确保 alpha 是非uint8
    assert alpha.dtype != np.uint8, "Alpha channel must not be of type uint8"

    h, w = alpha.shape

    # 1. 随机阈值化
    low = 0.01
    high = 1.0
    thres = random.random() * (high - low) + low
    mask = (alpha >= thres).astype(np.uint8)

    # 2. 随机进行腐蚀或膨胀操作
    random_num = random.randint(0, 3)
    # 确保核尺寸在预计算列表的范围内
    k_size = np.random.randint(min_kernel_size, min(max_kernel_size + 1, MAX_KERNEL_SIZE))
    kernel = EROSION_KERNELS[k_size]

    if random_num == 0:
        mask = cv2.erode(mask, kernel)
    elif random_num == 1:
        mask = cv2.dilate(mask, kernel)
    elif random_num == 2:
        mask = cv2.erode(mask, kernel)
        mask = cv2.dilate(mask, kernel)
    # random_num == 3: do nothing, or apply another combination
    elif random_num == 3:
        mask = cv2.dilate(mask, kernel)
        mask = cv2.erode(mask, kernel)

    # 3. 计算掩码的边界框坐标
    coords = np.argwhere(mask)
    if coords.size == 0:
        # 如果掩码为空，返回一个默认的小框
        mask_coords = np.array([0, 0, 1/w, 1/h])
    else:
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)
        
        # 归一化坐标
        x_min, x_max = x_min / w, x_max / w
        y_min, y_max = y_min / h, y_max / h
        mask_coords = np.array([x_min, y_min, x_max, y_max])

    return mask.astype(np.float32), mask_coords.astype(np.float32)


def gen_bbox(alpha: np.ndarray, coe_scale: float = 0.1) -> dict:
    """
    从 alpha 通道生成一个边界框 (bounding box).
    该过程会找到最大的连通分量，并在此基础上添加随机扰动.

    Args:
        alpha (np.ndarray): 输入的单通道 alpha 图像 (0.0-1.0, float).
        coe_scale (float): 控制边界框随机扰动范围的系数.

    Returns:
        dict: 一个包含 'bbox_mask' 和 'bbox_coords' 的字典.
              'bbox_mask': 边界框的二值掩码 (h, w), float32.
              'bbox_coords': 归一化的边界框坐标 [x_min, y_min, x_max, y_max], np.ndarray.
    """
    # 确保 alpha 是 float 类型
    assert alpha.dtype != np.uint8, "Alpha channel must be float."
        
    height, width = alpha.shape

    # 检查alpha是否几乎为空
    if np.count_nonzero(alpha) == 0:
        return np.zeros_like(alpha, dtype=np.float32), np.array([0, 0, 1/width, 1/height], dtype=np.float32)

    # 1. 找到最大的连通分量来确定主要物体
    binary_mask = alpha > 0
    labeled_array, num_features = label(binary_mask)
    
    # 默认使用整个alpha的边界
    y_coords, x_coords = np.where(binary_mask)
    y_min, x_min = y_coords.min(), x_coords.min()
    y_max, x_max = y_coords.max(), x_coords.max()

    if num_features > 1:
        # 计算每个连通区域的面积
        component_sizes = np.bincount(labeled_array.ravel())
        # 忽略背景（标签0）
        component_sizes = component_sizes[1:]
        if component_sizes.size > 0:
            # 找到最大的连通分量
            largest_component_label = np.argmax(component_sizes) + 1
            coords = np.argwhere(labeled_array == largest_component_label)
            y_min, x_min = coords.min(axis=0)
            y_max, x_max = coords.max(axis=0)

    # 2. 添加随机扰动
    coe = random.uniform(0, coe_scale)
    h_box, w_box = y_max - y_min, x_max - x_min
    
    padding_y = int(coe * h_box)
    padding_x = int(coe * w_box)
    
    # 随机决定是内缩还是外扩
    y_min_padded = y_min + random.choice([-1, 1]) * padding_y
    y_max_padded = y_max + random.choice([-1, 1]) * padding_y
    x_min_padded = x_min + random.choice([-1, 1]) * padding_x
    x_max_padded = x_max + random.choice([-1, 1]) * padding_x

    # 3. 确保坐标在图像范围内且合法 (min < max)
    y_min_final = min(y_min_padded, y_max_padded)
    y_max_final = max(y_min_padded, y_max_padded)
    x_min_final = min(x_min_padded, x_max_padded)
    x_max_final = max(x_min_padded, x_max_padded)
    
    y_min_final = max(0, y_min_final)
    y_max_final = min(height, y_max_final)
    x_min_final = max(0, x_min_final)
    x_max_final = min(width, x_max_final)
    
    # 4. 生成 bbox 掩码
    bbox_mask = np.zeros_like(alpha)
    bbox_mask[y_min_final:y_max_final, x_min_final:x_max_final] = 1

    # 5. 归一化坐标
    bbox_coords = np.array([
        x_min_final / width,
        y_min_final / height,
        x_max_final / width,
        y_max_final / height,
    ])

    return bbox_mask.astype(np.float32), bbox_coords.astype(np.float32)

def gen_points(alpha: np.ndarray, num_points: int = 10, psm: str = "gauss", radius: int = 20, thres: float = 0.8) -> dict:
    """
    从 alpha 通道中随机采样前景点，并生成对应的点掩码 (point mask).

    Args:
        alpha (np.ndarray): 输入的单通道 alpha 图像 (0.0-1.0, float).
        num_points (int): 要采样的点的数量.
        psm (str): 点扩散模型, 'gauss' 或 'circle'.
        radius (int): 'gauss' 的 sigma 或 'circle' 的半径.
        thres (float): 用于确定前景区域的阈值.

    Returns:
        dict: 一个包含 'point_mask' 和 'point_coords' 的字典.
              'point_mask': 所有采样点生成的掩码叠加图 (h, w), float32.
              'point_coords': 归一化的点坐标 [x1, y1, x2, y2, ...], np.ndarray.
    """
    # 确保 alpha 是 float 类型
    assert alpha.dtype != np.uint8, "Alpha channel must be float."
        
    height, width = alpha.shape
    
    # 1. 找到所有前景点
    y_coords, x_coords = np.where(alpha > thres)

    # 如果前景点不足，返回空结果
    if len(y_coords) < num_points:
        return np.zeros_like(alpha, dtype=np.float32), np.zeros(num_points * 2, dtype=np.float32)
    np.random.seed(42)
    # 2. 随机采样 N 个点
    indices = np.random.choice(len(y_coords), size=num_points, replace=False)
    selected_y = y_coords[indices]
    selected_x = x_coords[indices]
    
    point_mask = np.zeros_like(alpha, dtype=np.float32)
    point_coords = []
    
    # 3. 为每个点生成掩码并叠加
    for y_center, x_center in zip(selected_y, selected_x):
        tmp_mask = np.zeros_like(alpha, dtype=np.float32)
        if psm == "gauss":
            # 创建一个单点脉冲，然后进行高斯滤波
            tmp_mask[y_center, x_center] = 1
            tmp_mask = scipy.ndimage.gaussian_filter(tmp_mask, sigma=radius)
            # 归一化，使峰值为1
            if tmp_mask.max() > 0:
                tmp_mask /= tmp_mask.max()
        elif psm == "circle":
            # 使用 OpenCV 画一个实心圆，更高效
            cv2.circle(tmp_mask, (x_center, y_center), radius, 1, -1)

        # 将当前点的掩码合并到总掩码中
        point_mask = np.maximum(point_mask, tmp_mask)
        
        # 记录并归一化坐标
        point_coords.append(x_center / width)
        point_coords.append(y_center / height)

    return point_mask.astype(np.float32), np.array(point_coords, dtype=np.float32)

def get_true_intrinsics_from_hypersim(hypersim_matrix, width=768, height=768):
    """
    将 Hypersim 提供的 M_cam_from_uv 矩阵转换为标准的相机内参矩阵 K。

    Args:
        hypersim_matrix (torch.Tensor): [3, 3] 的 M_cam_from_uv 矩阵。
        width (int): 图像宽度。
        height (int): 图像高度。

    Returns:
        (torch.Tensor, torch.Tensor): 返回 (K, K_inv_true)，分别是 3x3 的标准内参和其逆。
    """
    device = hypersim_matrix.device
    dtype = hypersim_matrix.dtype
    
    # 构造 M_ndc_from_pix 矩阵
    # [x, y, 1] -> [u, v, 1]
    # u = (2*x / (W-1)) - 1
    # v = 1 - (2*y / (H-1))  (注意y轴翻转)
    M_ndc_from_pix = torch.tensor([
        [2.0 / (width - 1), 0.0, -1.0],
        [0.0, -2.0 / (height - 1), 1.0], # 负号用于翻转y轴
        [0.0, 0.0, 1.0]
    ], device=device, dtype=dtype)
    
    # 计算真正的 K_inv
    # K_inv_true 的作用是: P_cam = K_inv_true @ P_pixel
    K_inv_true = torch.matmul(hypersim_matrix, M_ndc_from_pix)
    
    # # K 就是 K_inv_true 的逆
    K = torch.inverse(K_inv_true)
    
    return K
class DataProcessingPipeline:
    def __init__(self, operators=None):
        self.operators: list[DataProcessingOperator] = [] if operators is None else operators
        
    def __call__(self, data):
        for operator in self.operators:
            data = operator(data)
        return data
    
    def __rshift__(self, pipe):
        if isinstance(pipe, DataProcessingOperator):
            pipe = DataProcessingPipeline([pipe])
        return DataProcessingPipeline(self.operators + pipe.operators)



class DataProcessingOperator:
    def __call__(self, data):
        raise NotImplementedError("DataProcessingOperator cannot be called directly.")
    
    def __rshift__(self, pipe):
        if isinstance(pipe, DataProcessingOperator):
            pipe = DataProcessingPipeline([pipe])
        return DataProcessingPipeline([self]).__rshift__(pipe)



class DataProcessingOperatorRaw(DataProcessingOperator):
    def __call__(self, data):
        return data



class ToInt(DataProcessingOperator):
    def __call__(self, data):
        return int(data)



class ToFloat(DataProcessingOperator):
    def __call__(self, data):
        return float(data)



class ToStr(DataProcessingOperator):
    def __init__(self, none_value=""):
        self.none_value = none_value
    
    def __call__(self, data):
        if data is None: data = self.none_value
        return str(data)



class LoadImage(DataProcessingOperator):
    def __init__(self, convert_RGB=False):
        self.convert_RGB = convert_RGB
    
    def __call__(self, data: str):
        if "depth" in data:
            # uint16 png
            image = Image.open(data)
        elif "normal" in data:
            # float32 3channel npy
            image = np.load(data)
        else:
            # uint8
            image = Image.open(data)
            if self.convert_RGB: image = image.convert("RGB")
        return image



class ImageCropAndResize(DataProcessingOperator):
    def __init__(self, height, width, max_pixels, height_division_factor, width_division_factor):
        self.height = height
        self.width = width
        self.max_pixels = max_pixels
        self.height_division_factor = height_division_factor
        self.width_division_factor = width_division_factor

    def crop_and_resize(self, image, target_height, target_width):
        width, height = image.size
        scale = max(target_width / width, target_height / height)
        image = torchvision.transforms.functional.resize(
            image,
            (round(height*scale), round(width*scale)),
            interpolation=torchvision.transforms.InterpolationMode.BILINEAR
        )
        image = torchvision.transforms.functional.center_crop(image, (target_height, target_width))
        return image
    
    def get_height_width(self, image):
        if self.height is None or self.width is None:
            width, height = image.size
            if width * height > self.max_pixels:
                scale = (width * height / self.max_pixels) ** 0.5
                height, width = int(height / scale), int(width / scale)
            height = height // self.height_division_factor * self.height_division_factor
            width = width // self.width_division_factor * self.width_division_factor
        else:
            height, width = self.height, self.width
        return height, width
    
    
    def __call__(self, data: Image.Image):
        image = self.crop_and_resize(data, *self.get_height_width(data))
        return image

class LoadImageToTensor(DataProcessingOperator):
    """加载图像并转换为tensor，支持RGB图像、深度图和法线图"""
    
    def __init__(self, convert_RGB=True,using_log=False,using_sqrt_disp=False,using_sqrt=False,using_pdf=False, pdf=None, with_mask=False, matting_prompt="trimap",height=768,width=768):
        self.convert_RGB = convert_RGB
        self.using_log = using_log
        self.using_sqrt = using_sqrt
        self.using_sqrt_disp = using_sqrt_disp
        self.using_pdf = using_pdf
        self.pdf = pdf
        self.with_mask = with_mask
        self.matting_prompt = matting_prompt
        self.height = height
        self.width = width


    def _load_image(self, data_path: str):
        """根据文件路径加载不同类型的图像"""

        if "Eval" in data_path or "rgb" in data_path or "img" in data_path or "matting" in data_path:
            image = Image.open(data_path)
            if self.convert_RGB:
                image = image.convert("RGB")
            if "kitti" in data_path.lower():
                # crop to 1216x352 for kitti depth evaluation
                _w, _h = image.size
                top_margin = int(_h - 352)
                left_margin = int((_w - 1216) / 2)
                image = image.crop((left_margin, top_margin, left_margin + 1216, top_margin + 352))
            return image
        elif "depth" in data_path:
            image = Image.open(data_path)
            if "kitti" in data_path.lower():
                # crop to 1216x352 for kitti depth evaluation
                _w, _h = image.size
                top_margin = int(_h - 352)
                left_margin = int((_w - 1216) / 2)
                image = image.crop((left_margin, top_margin, left_margin + 1216, top_margin + 352))
            return image
        elif "normal" in data_path:
            return np.load(data_path)
        else:
            image = Image.open(data_path)
            if self.convert_RGB:
                image = image.convert("RGB")
            return image
    def _to_tensor(self, image: Union[Image.Image,np.ndarray], data_path: str):
        """将图像转换为tensor并标准化"""
        if "Eval" in data_path or "rgb" in data_path or "img" in data_path or "matting" in data_path:
            # uint8 for Eval input
            image = image.resize((self.width,self.height),Image.Resampling.BILINEAR)
            tensor = TF.to_tensor(image) * 2 -1
        elif "depth" in data_path:
            # 处理深度图
            image = image.resize((self.width,self.height))
            array = np.array(image,dtype=np.float32)
            if "vkitti" in data_path:
                array = array/100.0
                mask = np.logical_and(array>1e-1,array<80.0)
            elif "hyp" in data_path.lower():
                array = array/1000.0
                mask = np.logical_and(array>1e-3,array<65.5)
            if self.using_log:
                array = np.log(array + 1e-6)
            elif self.using_sqrt_disp:
                array = 1/np.sqrt(array+1e-6)
            elif self.using_sqrt:
                array = np.sqrt(array+1e-6)
            elif self.using_pdf:
                array = np.interp(array, self.pdf['bins'], self.pdf['y_map'])
            # keep in 2-98 percentile, scale to [-1, 1]
            p2, p98 = np.percentile(array[mask], (2, 98))
            # if p98-p2<1: return None 
            
            if p98 - p2 < 1e-3:
                print(f"Warning: {data_path} has invalid depth values.")
                with open("error_depth_Hyp_Log.txt","a") as f:
                    f.write(f"{data_path}\n")
            array = (array - p2) / (p98 - p2)
            array = (array - 0.5) * 2
            array = np.clip(array, -1, 1)
            
            tensor = torch.from_numpy(array).unsqueeze(0).repeat(3, 1, 1)
        elif "normal" in data_path:
            image[:,:,0] *= -1
            image = cv2.resize(image, (self.width,self.height), interpolation=cv2.INTER_LINEAR_EXACT)
            image = image.astype(np.float32)
            # already a ndarray in scale [-1, 1]
            tensor = torch.from_numpy(image).permute(2, 0, 1)
                # --- 归一化 ---
            norm = torch.norm(tensor, dim=0, keepdim=True) + 1e-8
            tensor = tensor / norm

            mask = norm.squeeze(0) > 1e-3
        else:
            image = image.resize((self.width,self.height),Image.Resampling.BILINEAR)
            # tensor = torch.from_numpy(np.array(image,dtype=np.float32))
            # tensor = tensor / 255.0 * 2 - 1
            # tensor = tensor.permute(2, 0, 1)
            tensor = TF.to_tensor(image) * 2 -1
        
        
        if self.with_mask and "rgb" not in data_path and "img" not in data_path:
            # final tensor shape (C+1,H,W)
            # first C channels for image, scale [-1,1] dtype float32
            # last channel for mask, scale [0,1], dtype float32 (in fact a bool)
            if "depth" in data_path:
                mask = (mask).astype(np.float32)
                mask = torch.from_numpy(mask).unsqueeze(0)
                tensor = torch.cat([tensor, mask], dim=0)
            elif "normal" in data_path:
                mask = mask.float().unsqueeze(0)
                tensor = torch.cat([tensor, mask], dim=0)
            elif "mask" in data_path or "alpha" in data_path:
                # trimap as a mask
                alpha = (tensor[0].numpy()+1)/2 # 3channels are same, so use first channel, 0~1 float, hxw
                visual_prompt, _ = gen_trimap((alpha*255).astype(np.uint8))
                visual_prompt = torch.from_numpy(visual_prompt).unsqueeze(0).to(tensor.dtype)
                tensor = torch.cat([tensor, visual_prompt], dim=0)

        return tensor
    
    def __call__(self, data_path: str):
        image = self._load_image(data_path)
        tensor = self._to_tensor(image, data_path)
        return tensor

class LoadPILImageTensor(DataProcessingOperator):
    ## 加载已经是PIL Image格式的图像，并转换为tensor
    def __init__(self, convert_RGB=True,height=768,width=768):
        self.convert_RGB = convert_RGB
        self.height = height
        self.width = width
    def __call__(self, image: Image.Image):
        if self.convert_RGB:
            image = image.convert("RGB")
        image = image.resize((self.width,self.height),Image.Resampling.BILINEAR)
        tensor = TF.to_tensor(image) * 2 -1
        return tensor
class TensorCropAndResize(DataProcessingOperator):
    """在tensor上进行裁剪和缩放操作"""
    
    def __init__(self, height=None, width=None, max_pixels=None, 
                 height_division_factor=1, width_division_factor=1):
        self.height = height
        self.width = width
        self.max_pixels = max_pixels
        self.height_division_factor = height_division_factor
        self.width_division_factor = width_division_factor
    
    def _calculate_target_size(self, tensor):
        """计算目标尺寸"""
        _, height, width = tensor.shape
        
        if self.height is None or self.width is None:
            # 自动计算尺寸
            if self.max_pixels and width * height > self.max_pixels:
                scale = (width * height / self.max_pixels) ** 0.5
                height, width = int(height / scale), int(width / scale)
            
            # 确保能被division_factor整除
            height = height // self.height_division_factor * self.height_division_factor
            width = width // self.width_division_factor * self.width_division_factor
        else:
            # 使用指定尺寸
            height, width = self.height, self.width
            
        return height, width
    
    def _crop_and_resize_tensor(self, tensor, target_height, target_width):
        """对tensor进行缩放和中心裁剪"""
        # 添加batch维度进行操作
        tensor = tensor.unsqueeze(0)  # (C,H,W) -> (1,C,H,W)
        
        # 计算缩放比例（保持长宽比，确保目标尺寸能完全包含）
        _, _, current_height, current_width = tensor.shape
        scale = max(target_width / current_width, target_height / current_height)
        
        # 缩放
        new_height = round(current_height * scale)
        new_width = round(current_width * scale)
        
        tensor = F.interpolate(
            tensor, 
            size=(new_height, new_width), 
            mode='bicubic', 
            align_corners=True
        )
        
        # 中心裁剪
        _, _, scaled_height, scaled_width = tensor.shape
        top = (scaled_height - target_height) // 2
        left = (scaled_width - target_width) // 2
        
        tensor = tensor[:, :, top:top+target_height, left:left+target_width]
        
        # 移除batch维度
        tensor = tensor.squeeze(0)  # (1,C,H,W) -> (C,H,W)
        
        return tensor
    
    def __call__(self, tensor):
        """
        Args:
            tensor: 形状为(C,H,W)的tensor
        Returns:
            处理后的tensor，形状为(C,target_H,target_W)
        """
        target_height, target_width = self._calculate_target_size(tensor)
        processed_tensor = self._crop_and_resize_tensor(tensor, target_height, target_width)
        return processed_tensor


class ToList(DataProcessingOperator):
    def __call__(self, data):
        return [data]
    


# class LoadVideo(DataProcessingOperator):
#     def __init__(self, num_frames=81, time_division_factor=4, time_division_remainder=1, frame_processor=lambda x: x):
#         self.num_frames = num_frames
#         self.time_division_factor = time_division_factor
#         self.time_division_remainder = time_division_remainder
#         # frame_processor is build in the video loader for high efficiency.
#         self.frame_processor = frame_processor
        
#     def get_num_frames(self, reader):
#         num_frames = self.num_frames
#         if int(reader.count_frames()) < num_frames:
#             num_frames = int(reader.count_frames())
#             while num_frames > 1 and num_frames % self.time_division_factor != self.time_division_remainder:
#                 num_frames -= 1
#         return num_frames
        
#     def __call__(self, data: str):
#         reader = imageio.get_reader(data)
#         num_frames = self.get_num_frames(reader)
#         frames = []
#         for frame_id in range(num_frames):
#             frame = reader.get_data(frame_id)
#             frame = Image.fromarray(frame)
#             frame = self.frame_processor(frame)
#             frames.append(frame)
#         reader.close()
#         return frames



class SequencialProcess(DataProcessingOperator):
    def __init__(self, operator=lambda x: x):
        self.operator = operator
        
    def __call__(self, data):
        return [self.operator(i) for i in data]



# class LoadGIF(DataProcessingOperator):
#     def __init__(self, num_frames=81, time_division_factor=4, time_division_remainder=1, frame_processor=lambda x: x):
#         self.num_frames = num_frames
#         self.time_division_factor = time_division_factor
#         self.time_division_remainder = time_division_remainder
#         # frame_processor is build in the video loader for high efficiency.
#         self.frame_processor = frame_processor
        
#     def get_num_frames(self, path):
#         num_frames = self.num_frames
#         images = iio.imread(path, mode="RGB")
#         if len(images) < num_frames:
#             num_frames = len(images)
#             while num_frames > 1 and num_frames % self.time_division_factor != self.time_division_remainder:
#                 num_frames -= 1
#         return num_frames
        
#     def __call__(self, data: str):
#         num_frames = self.get_num_frames(data)
#         frames = []
#         images = iio.imread(data, mode="RGB")
#         for img in images:
#             frame = Image.fromarray(img)
#             frame = self.frame_processor(frame)
#             frames.append(frame)
#             if len(frames) >= num_frames:
#                 break
#         return frames
    


class RouteByExtensionName(DataProcessingOperator):
    def __init__(self, operator_map):
        self.operator_map = operator_map
        
    def __call__(self, data: str):
        file_ext_name = data.split(".")[-1].lower()
        for ext_names, operator in self.operator_map:
            if ext_names is None or file_ext_name in ext_names:
                return operator(data)
        raise ValueError(f"Unsupported file: {data}")



class RouteByType(DataProcessingOperator):
    def __init__(self, operator_map):
        self.operator_map = operator_map
        
    def __call__(self, data):
        for dtype, operator in self.operator_map:
            if dtype is None or isinstance(data, dtype):
                return operator(data)
        raise ValueError(f"Unsupported data: {data}")



class LoadTorchPickle(DataProcessingOperator):
    def __init__(self, map_location="cpu"):
        self.map_location = map_location
        
    def __call__(self, data):
        return torch.load(data, map_location=self.map_location, weights_only=False)



class ToAbsolutePath(DataProcessingOperator):
    def __init__(self, base_path=""):
        self.base_path = base_path
        
    def __call__(self, data):
        # If data is already an absolute path or base_path is empty/None, return data as is
        if os.path.isabs(data) or not self.base_path:
            return data
        return os.path.join(self.base_path, data)



class UnifiedDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        base_path=None, metadata_path=None,
        repeat=1,
        data_file_keys=tuple(),
        main_data_operator=lambda x: x,
        special_operator_map=None,
        default_caption=None,
        matting_prompt=None,
        use_coor_input=False,
        use_camera_intrinsics=False,
    ):
        self.base_path = base_path
        self.metadata_path = metadata_path
        self.repeat = repeat
        self.data_file_keys = data_file_keys
        self.main_data_operator = main_data_operator
        self.cached_data_operator = LoadTorchPickle()
        self.special_operator_map = {} if special_operator_map is None else special_operator_map
        self.data = []
        self.cached_data = []
        self.load_from_cache = metadata_path is None
        self.default_caption = default_caption
        self.matting_prompt = matting_prompt
        self.use_coor_input = use_coor_input
        self.use_camera_intrinsics = use_camera_intrinsics
        self.load_metadata(metadata_path)
        if use_camera_intrinsics:
            if "vkitti" in base_path.lower():
                # K[0,0] K[1,1] K[0,2] K[1,2] = 725.0087, 725.0087, 620.5, 187.0
                # make a 3x3 torch tensor
                # self.camera_intrinsics = torch.tensor([[725.0087, 0.0, 620.5],
                #                                        [0.0, 725.0087, 187.0],
                #                                        [0.0, 0.0, 1.0]], dtype=torch.float32)
                self.camera_intrinsics = torch.tensor([[ 448.3300,    0.0000,  383.6901],
                                                        [   0.0000, 1484.8178,  382.9760],
                                                        [   0.0000,    0.0000,    1.0000]])
            elif "hyp" in base_path.lower():
                path = "../dataset/Hypersim/processed_depth/metadata_camera_parameters.csv"
                df = pandas.read_csv(path)
                columns = ['scene_name', 'M_cam_from_uv_00', 'M_cam_from_uv_01', 'M_cam_from_uv_02',
                        'M_cam_from_uv_10', 'M_cam_from_uv_11', 'M_cam_from_uv_12',
                        'M_cam_from_uv_20', 'M_cam_from_uv_21', 'M_cam_from_uv_22']
                K = lambda row: [[float(row[columns[1]]), float(row[columns[2]]), float(row[columns[3]])],
                    [float(row[columns[4]]), float(row[columns[5]]), float(row[columns[6]])],
                    [float(row[columns[7]]), float(row[columns[8]]), float(row[columns[9]])]]
                self.camera_intrinsics = {}
                for i in range(len(df)):
                    row = df.iloc[i]
                    scene_name = row['scene_name']
                    hypersim_matrix = torch.tensor(K(row), dtype=torch.float32)
                    self.camera_intrinsics[scene_name] = get_true_intrinsics_from_hypersim(hypersim_matrix, width=768, height=768)

    @staticmethod
    def default_image_operator(
        base_path="",
        max_pixels=1024*1024, height=768, width=768,
        height_division_factor=16, width_division_factor=16,
        using_log=False,
        using_sqrt=False,
        using_sqrt_disp=False,
        with_mask=False,
        using_pdf=False, 
        pdf=None, 
    ):
        # Handle None base_path
        if base_path is None:
            base_path = ""
            
        return RouteByType(operator_map=[
            (str, ToAbsolutePath(base_path) >> LoadImageToTensor(using_log=using_log,using_sqrt_disp=using_sqrt_disp, using_sqrt=using_sqrt, using_pdf=using_pdf, pdf=pdf, with_mask=with_mask,height=height,width=width)),
            (list, SequencialProcess(ToAbsolutePath(base_path) >> LoadImageToTensor(using_log=using_log,using_sqrt_disp=using_sqrt_disp,using_sqrt=using_sqrt, using_pdf=using_pdf, pdf=pdf, with_mask=with_mask,height=height,width=width))),
            (Image.Image, LoadPILImageTensor(convert_RGB=True,height=height,width=width)),
        ])
    
    # @staticmethod
    # def default_video_operator(
    #     base_path="",
    #     max_pixels=1920*1080, height=None, width=None,
    #     height_division_factor=16, width_division_factor=16,
    #     num_frames=81, time_division_factor=4, time_division_remainder=1,
    # ):
    #     # Handle None base_path
    #     if base_path is None:
    #         base_path = ""
            
    #     return RouteByType(operator_map=[
    #         (str, ToAbsolutePath(base_path) >> RouteByExtensionName(operator_map=[
    #             (("jpg", "jpeg", "png", "webp"), LoadImage() >> ImageCropAndResize(height, width, max_pixels, height_division_factor, width_division_factor) >> ToList()),
    #             (("gif",), LoadGIF(num_frames, time_division_factor, time_division_remainder) >> ImageCropAndResize(height, width, max_pixels, height_division_factor, width_division_factor)),
    #             (("mp4", "avi", "mov", "wmv", "mkv", "flv", "webm"), LoadVideo(
    #                 num_frames, time_division_factor, time_division_remainder,
    #                 frame_processor=ImageCropAndResize(height, width, max_pixels, height_division_factor, width_division_factor),
    #             )),
    #         ])),
    #     ])
        
    def search_for_cached_data_files(self, path):
        for file_name in os.listdir(path):
            subpath = os.path.join(path, file_name)
            if os.path.isdir(subpath):
                self.search_for_cached_data_files(subpath)
            elif subpath.endswith(".pth"):
                self.cached_data.append(subpath)
    
    def load_metadata(self, metadata_path):
        if metadata_path is None:
            print("No metadata_path. Searching for cached data files.")
            self.search_for_cached_data_files(self.base_path)
            print(f"{len(self.cached_data)} cached data files found.")
        elif metadata_path.endswith(".json"):
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            self.data = metadata
        elif metadata_path.endswith(".jsonl"):
            metadata = []
            with open(metadata_path, 'r') as f:
                for line in f:
                    metadata.append(json.loads(line.strip()))
            self.data = metadata
        elif metadata_path.endswith(".csv"):
            metadata = pandas.read_csv(metadata_path)
            self.camera_intrinsics = [metadata.iloc[i].to_dict() for i in range(len(metadata))]
        elif metadata_path.endswith(".txt"):
            metadata = []
            # with some columns named as self.data_file_keys, seprated by space
            with open(metadata_path, 'r') as f:
                lines = f.readlines()
            # assert len(self.data_file_keys)==len(lines[0].strip().split()), f"Number of columns in txt file should be equal to number of data_file_keys."
            for line in lines:
                if "\t" in line:
                    items = line.strip().split("\t")
                else:
                    items = line.strip().split()
                item_dict = {k:v for k,v in zip(self.data_file_keys, items)}
                metadata.append(item_dict)
            self.data = metadata

    def __getitem__(self, data_id):
        if self.load_from_cache:
            data = self.cached_data[data_id % len(self.cached_data)]
            data = self.cached_data_operator(data)
        else:
            data = self.data[data_id % len(self.data)].copy()
            # for key in self.data_file_keys:
            #     if key in data:
            #         if key in self.special_operator_map:
            #             data[key] = self.special_operator_map[key]
            #         elif key in self.data_file_keys:
            #             data[key] = self.main_data_operator(data[key])
            if self.use_camera_intrinsics:
                if isinstance(self.camera_intrinsics, dict):
                    parts = data[self.data_file_keys[0]].split("/")
                    scene_name = parts[0] if len(parts)==2 else parts[1]
                    data["camera_intrinsics"] = self.camera_intrinsics[scene_name]
                else:
                    data["camera_intrinsics"] = self.camera_intrinsics
                data["input_depth"] = Image.open(os.path.join(self.base_path, data[self.data_file_keys[1]])).resize((768,576))
                if "vkitti" in self.base_path.lower():
                    data["input_depth"] = torch.from_numpy((np.array(data["input_depth"]))/100.0).to(torch.bfloat16)
                elif "hyp" in self.base_path.lower():
                    data["input_depth"] = torch.from_numpy((np.array(data["input_depth"]))/1000.0).to(torch.bfloat16)
            for key in self.data_file_keys:
                if key in data:
                    data[key] = self.main_data_operator(data[key]).to(torch.bfloat16)
            for key in self.special_operator_map:
                if key=="mask":
                    _tmp = data["image"].clone()
                    data["image"] = _tmp[:3] # first 3 channels for image
                    data["mask"] = _tmp[3] # last channel for mask
                    if "matting" in self.metadata_path:
                        # for matting dataset, mask should be bool
                        data["trimap"] = data["mask"].clone() # 0~255 uint8
                        alpha = (data["image"][0].clone().to(torch.float32).cpu().numpy() + 1) / 2.0 # 0~1 float array

                        visual_prompt_coords = None
                        if self.matting_prompt is not None:
                            if self.matting_prompt=="trimap":
                                # visual_prompt = (data["mask"].to(torch.bfloat16)/255.0) # 0~1 float tensor
                                visual_prompt, visual_prompt_coords = gen_trimap((alpha*255).astype(np.uint8))
                                visual_prompt = (visual_prompt/255.0).astype(np.float32) # 0~1 float array
                            elif self.matting_prompt=="mask":
                                visual_prompt, visual_prompt_coords = gen_mask(alpha)
                            elif self.matting_prompt=="bbox":
                                visual_prompt, visual_prompt_coords = gen_bbox(alpha,0.01)
                            elif self.matting_prompt=="points":
                                visual_prompt, visual_prompt_coords = gen_points(alpha)
                            if isinstance(visual_prompt, np.ndarray):
                                visual_prompt = torch.from_numpy(visual_prompt).unsqueeze(0).to(data["image"].dtype) # 1~1 float tensor
                            visual_prompt  = (visual_prompt*2-1).repeat(3,1,1)
                            data["kontext_images"] = [data["kontext_images"], visual_prompt]
                            # scale to 1/8 size of original image
                            # data["kontext_masks"] = torch.nn.functional.interpolate(visual_prompt.unsqueeze(0), scale_factor=0.125, mode='nearest').squeeze(0)
                            del visual_prompt
                        data["mask"] = (data["mask"]==128)
                        if visual_prompt_coords is not None and self.use_coor_input:
                            data["visual_prompt_coords"] = torch.from_numpy(visual_prompt_coords).to(torch.float32)
                        del alpha
                    # print(data['mask'].shape,data['mask'].dtype,data['mask'].min(),data['mask'].max())
                    del _tmp
                elif key=="prompt":
                    data["prompt"] = self.default_caption
                
        return data

    def __len__(self):
        if self.load_from_cache:
            return len(self.cached_data) * self.repeat
        else:
            return len(self.data) * self.repeat
        
    def check_data_equal(self, data1, data2):
        # Debug only
        if len(data1) != len(data2):
            return False
        for k in data1:
            if data1[k] != data2[k]:
                return False
        return True
