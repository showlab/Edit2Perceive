import argparse
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional

import torch
import numpy as np
from tqdm import tqdm
from PIL import Image

from diffsynth.pipelines.flux_image_new import FluxImagePipeline, ModelConfig
from diffsynth import load_state_dict
from diffsynth.trainers.unified_dataset import UnifiedDataset, gen_mask, gen_bbox, gen_points, gen_trimap
from diffsynth.utils.eval_multiple_datasets import parse_flux_model_configs
import matplotlib.pyplot as plt
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_root", type=str, default="/mnt/nfs/share_model/FLUX.1-Kontext-dev", help="Flux model root directory")
    parser.add_argument("--state-dict", type=str, default=f"models/train/kontext/bs64_log_cons/step-4350.safetensors", help="训练好的 state_dict path to load")
    # parser.add_argument("--state_dict", type=str, default=f"models/train/kontext/bs64_sqrt_cons/step-5300.safetensors", help="训练好的 state_dict path to load")
    # parser.add_argument("--state-dict", type=str, default=f"models/train/kontext_normal/bs16_cons/step-22620.safetensors", help="训练好的 state_dict path to load")
    # parser.add_argument("--state-dict", type=str, default=f"models/train/kontext_matting/bs16_cons_mixSDMatte_points/step-5200.safetensors", help="训练好的 state_dict path to load")
    # parser.add_argument("--state-dict", type=str, default=f"models/train/kontext_matting/bs16_cons_mixSDMatte_empty/step-4100.safetensors", help="训练好的 state_dict path to load")
    parser.add_argument("--input_path", type=str, default="visualization/teaser/flower1.png,visualization/teaser/flower2.png", help="Path to the input image")
    parser.add_argument("--output_path", type=str, default=None, help="Path to the output image")
    parser.add_argument("--task", type=str, choices=["depth", "normal", "matting"], help="Task to perform")
    resolution = 1536
    args = parser.parse_args()
    torch_dtype = torch.bfloat16

    if "normal" in args.state_dict:
        print("Detected normal state dict")
        args.task = "normal"
    elif "matting" in args.state_dict:
        print("Detected matting state dict")
        args.task = "matting"
    else:
        print("Detected depth state dict")
        args.task = "depth"
    
    print("Loading FluxImagePipeline ...")
    pipe = FluxImagePipeline.from_pretrained(
        torch_dtype=torch_dtype,
        device="cuda",
        model_configs=parse_flux_model_configs(args.model_root)
    )
    state_dict = load_state_dict(args.state_dict)
    pipe.dit.load_state_dict(state_dict)
    transform = UnifiedDataset.default_image_operator(height=resolution, width=resolution)
    if "," in args.input_path:
        images = args.input_path.split(",")
    else:
        images = [args.input_path]
    for image_path in images:
        if args.task == "depth" or args.task == "normal":
            out_np = pipe(
                prompt=f"Transform to {args.task} map while maintaining original composition",
                kontext_images=transform(image_path),
                height=768, width=768,
                embedded_guidance=1,
                num_inference_steps=8,
                seed=42,
                output_type="np",
                rand_device="cuda",
                task=args.task,
            )
        elif args.task == "matting":
            # points visual prompt need to generate first
            # let alpha be the center crop of the image
            alpha = np.zeros((resolution, resolution), dtype=np.float32)
            alpha[resolution//4:resolution*3//4, resolution//4:resolution*3//4] = 1.0
            points,coords = gen_points(alpha, num_points=10,radius=30)
            img = Image.open(image_path).convert("RGB").resize((resolution, resolution))
            # 在原图上绘制一下点并保存
            # coords是长度为20的np.ndarray，
            for i in range(coords.shape[0]//2):
                x, y = coords[i*2], coords[i*2+1]
                # img.putpixel((int(x*resolution), int(y*resolution)), (255, 0, 0))
                # put a circle
                for dx in range(-9, 10):
                    for dy in range(-9, 10):
                        if dx*dx + dy*dy <= 81:
                            if 0 <= int(x*resolution)+dx < resolution and 0 <= int(y*resolution)+dy < resolution:
                                img.putpixel((int(x*resolution)+dx, int(y*resolution)+dy), (255, 0, 0))
            img.save(image_path.replace(".jpg",".png").replace(".png", "_debug_points.png"))
            points = torch.from_numpy(points*2-1).repeat(3,1,1).to("cuda")
            out_np = pipe(
                prompt=f"Extract the foreground object and generate alpha matte",
                kontext_images=[transform(image_path), points],
                height=768, width=768,
                embedded_guidance=1,
                num_inference_steps=10,
                seed=42,
                output_type="np",
                rand_device="cuda",
                task=args.task,
            )
        if args.task == "depth":
            if out_np.ndim == 3:
                out_np = np.mean(out_np, axis=2)
            cmap = plt.get_cmap('Spectral')
            out_np = cmap(out_np)[:, :, :3]  # remove alpha channel
            out_np = (out_np * 255).astype(np.uint8)
        elif args.task == "normal":
            out_np = (out_np + 1) / 2 * 255.0
            out_np = out_np.astype(np.uint8)
        elif args.task == "matting":
            out_np = ((out_np>0.5) * 255.0).astype(np.uint8)
        if args.output_path is None:
            output_path = image_path.replace(".jpg",".png").replace(".png", f"_{args.task}.png")
        else:
            output_path = args.output_path
        Image.fromarray(out_np).save(output_path)
        print(f"Saved output to {output_path}")