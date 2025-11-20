import numpy as np
import matplotlib.pyplot as plt
import torch
def prepare_image(tensor):
    """把tensor处理成matplotlib能显示的格式 (H,W,C) or (H,W)"""
    if torch.is_tensor(tensor):
        tensor = tensor.detach().cpu()
    # 如果是 (C,H,W)，转为 (H,W,C)
    # 如果是 (B,C,H,W)，取第一个样本，转为 (H,W,C)
    if tensor.ndim == 4:
        tensor = tensor[0]
    if tensor.ndim == 3 and tensor.shape[0]>3:
        # (C,H,W) but C>3, 取前3通道
        tensor = tensor[:3]
    if tensor.ndim == 3:
        # 归一化 -1~1 -> 0~1
        if tensor.min() < -0.1 and tensor.max() > 1.1:
            tensor = (tensor + 1) / 2
            tensor = tensor.clamp(0, 1)
        tensor = tensor.permute(1, 2, 0)  # (H,W,C)
    elif tensor.ndim == 2:
        # mask, 0~1之间
        if tensor.min() < 0.05 and tensor.min()>-0.05:
            tensor = tensor.clamp(0, 1)
        tensor = tensor
    return (tensor.numpy()*255).astype(np.uint8)
def visualize_sample(sample, figsize=(12, 4)):
    """
    智能可视化函数
    sample: dict, 包含 'kontext_images', 'image', 'mask'
    """

    kontext = prepare_image(sample["kontext_images"].to(torch.float32))
    image   = prepare_image(sample["image"].to(torch.float32))
    mask    = prepare_image(sample["mask"].to(torch.float32)) if "mask" in sample else None
    print(f"kontext: {type(kontext)}, shape: {kontext.shape}, dtype: {kontext.dtype}")
    print(f"image: {type(image)}, shape: {image.shape}, dtype: {image.dtype}")

    fig, axs = plt.subplots(1, 3, figsize=figsize)
    axs[0].imshow(kontext)
    axs[0].set_title("kontext_images")
    axs[0].axis("off")

    axs[1].imshow(image)
    axs[1].set_title("image")
    axs[1].axis("off")

    axs[2].imshow((mask).astype(np.uint8), cmap="gray")
    axs[2].set_title("mask")
    axs[2].axis("off")

    plt.tight_layout()
    plt.savefig("tmp.png")

