import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from typing import Optional, Sequence, Union

import torch
import numpy as np
from PIL import Image

from pipelines.flux_image_new import FluxImagePipeline
from models.utils import load_state_dict
from trainers.unified_dataset import UnifiedDataset, gen_points
from utils.eval_multiple_datasets import parse_flux_model_configs
import matplotlib.pyplot as plt

# 模型配置
MODEL_CONFIGS = {
    "depth": "ckpts/edit2percieve_depth.safetensors",
    "normal": "ckpts/edit2percieve_normal.safetensors",
    "matting": "ckpts/edit2percieve_matting.safetensors",
}

def inference(
    model_root: str,
    task: str,
    input_paths: Union[str, Sequence[str]],
    resolution: int = 768,
    num_inference_steps: int = 8,
    seed: int = 42,
    output_path: Optional[str] = None,
    torch_dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
) -> None:
    """Run inference on one or more images."""
    
    # 获取模型路径
    if task not in MODEL_CONFIGS:
        raise ValueError(f"task must be one of {list(MODEL_CONFIGS.keys())}")
    
    state_dict_path = MODEL_CONFIGS[task]
    
    # 加载模型
    print("Loading FluxImagePipeline ...")
    pipe = FluxImagePipeline.from_pretrained(
        torch_dtype=torch_dtype,
        device=device,
        model_configs=parse_flux_model_configs(model_root),
    )
    state_dict = load_state_dict(state_dict_path)
    pipe.dit.load_state_dict(state_dict)
    print("Model loaded successfully!")
    
    # 准备图片列表
    if isinstance(input_paths, str):
        images = [path for path in input_paths.split(",") if path]
    else:
        images = list(input_paths)
    
    transform = UnifiedDataset.default_image_operator(height=resolution, width=resolution)
    
    # 处理每张图片
    for image_path in images:
        if task in {"depth", "normal"}:
            out_np = pipe(
                prompt=f"Transform to {task} map while maintaining original composition",
                kontext_images=transform(image_path),
                height=768,
                width=768,
                embedded_guidance=1,
                num_inference_steps=num_inference_steps,
                seed=seed,
                output_type="np",
                rand_device=device,
                task=task,
            )
        else:  # matting
            alpha = np.zeros((resolution, resolution), dtype=np.float32)
            alpha[resolution // 4 : resolution * 3 // 4, resolution // 4 : resolution * 3 // 4] = 1.0
            points, coords = gen_points(alpha, num_points=10, radius=30)
            img = Image.open(image_path).convert("RGB").resize((resolution, resolution))
            for i in range(coords.shape[0] // 2):
                x, y = coords[i * 2], coords[i * 2 + 1]
                for dx in range(-9, 10):
                    for dy in range(-9, 10):
                        if dx * dx + dy * dy <= 81:
                            px = int(x * resolution) + dx
                            py = int(y * resolution) + dy
                            if 0 <= px < resolution and 0 <= py < resolution:
                                img.putpixel((px, py), (255, 0, 0))
            img.save(image_path.replace(".jpg", ".png").replace(".png", "_debug_points.png"))
            points = torch.from_numpy(points * 2 - 1).repeat(3, 1, 1).to(device)
            out_np = pipe(
                prompt="Extract the foreground object and generate alpha matte",
                kontext_images=[transform(image_path), points],
                height=768,
                width=768,
                embedded_guidance=1,
                num_inference_steps=10,
                seed=seed,
                output_type="np",
                rand_device=device,
                task=task,
            )
        
        # 后处理
        if task == "depth":
            if out_np.ndim == 3:
                out_np = np.mean(out_np, axis=2)
            cmap = plt.get_cmap("Spectral")
            out_np = cmap(out_np)[:, :, :3]
            out_np = (out_np * 255).astype(np.uint8)
        elif task == "normal":
            out_np = (out_np + 1) / 2 * 255.0
            out_np = out_np.astype(np.uint8)
        else:  # matting
            out_np = ((out_np > 0.5) * 255.0).astype(np.uint8)
        
        # 保存结果
        if output_path is None:
            save_path = image_path.replace(".jpg", ".png").replace(".png", f"_{task}.png")
        else:
            save_path = output_path
        
        Image.fromarray(out_np).save(save_path)
        print(f"Saved output to {save_path}")


if __name__ == "__main__":
    # Please Change the model root path below to your own model directory
    model_root = "/mnt/nfs/share_model/FLUX.1-Kontext-dev"
    
    inference(
        model_root=model_root,
        task="depth",  # Options: "depth", "normal", "matting"
        input_paths="samples/cutecat.jpg"
    )