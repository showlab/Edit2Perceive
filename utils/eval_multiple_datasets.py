#!/usr/bin/env python3
"""
简化版：一次评估多个数据集，支持批量推理

用法示例：
    python eval_multiple_datasets.py \
        --model-root ./FLUX.1-Kontext-dev \
        --state-dict models/train/kontext/bs64_mask/step-3200.safetensors \
        --datasets scannet,nyuv2 \
        --max-samples 800 \
        --batch-size 4

或者不传 --datasets 则评估 DATASETS 中列出的所有数据集（按顺序）。
"""

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

from pipelines.flux_image_new import FluxImagePipeline, ModelConfig
from models.utils import load_state_dict, parse_flux_model_configs
from models.unified_dataset import UnifiedDataset, gen_mask, gen_bbox, gen_points, gen_trimap
from utils.eval_depth import test as eval_depth
from utils.eval_normal import test as eval_normal
from utils.eval_matting import test as eval_matting

@dataclass
class DatasetConfig:
    name: str
    file_list: str        # 文件列表，每行一般是 "rel_path [other cols]" 或单列路径
    gt_path: str          # 用于评估的 ground-truth 根目录（传给 eval_depth）
    output_dir: str       # 存放预测 npy 的目录（pred_path）
    dataset_arg: str      # 传给 eval_depth 的 dataset 名字（如 "scannet","nyuv2","kitti","eth3d"）
    # 可以按需扩展更多 eval 参数


# ====== 在这里添加/修改你要评估的数据集 ======
DATASETS_DEPTH: List[DatasetConfig] = [
    DatasetConfig(
        name="nyuv2",
        file_list="./data_split/nyu_depth/labeled/filename_list_test.txt",
        gt_path="/mnt/nfs/workspace/syq/dataset/Eval/depth/nyuv2/",
        output_dir="result/nyuv2/",
        dataset_arg="nyu",
    ),
    DatasetConfig(
        name="kitti",
        file_list="./data_split/kitti_depth/eigen_test_files_with_gt.txt",
        gt_path="/mnt/nfs/workspace/syq/dataset/Eval/depth/kitti/",
        output_dir="result/kitti/",
        dataset_arg="kitti",
    ),
    DatasetConfig(
        name="eth3d",
        file_list="./data_split/eth3d_depth/eth3d_filename_list.txt",
        gt_path="/mnt/nfs/workspace/syq/dataset/Eval/depth/eth3d/",
        output_dir="result/eth3d/",
        dataset_arg="eth3d",
    ),
    DatasetConfig(
        name="scannet",
        file_list="./data_split/scannet_depth/scannet_val_sampled_list_800_1.txt",
        gt_path="/mnt/nfs/workspace/syq/dataset/Eval/depth/scannet",
        output_dir="result/scannet/",
        dataset_arg="scannet",
    ),
    DatasetConfig(
        name="diode",
        file_list="./data_split/diode_depth/diode_val_all_filename_list.txt",
        gt_path="/mnt/nfs/workspace/syq/dataset/Eval/depth/diode/",
        output_dir="result/diode/",
        dataset_arg="diode",
    ),
]
DATASETS_NORMAL: List[DatasetConfig] = [
    DatasetConfig(
        name="nyuv2",
        file_list="./data_split/nyu_normals/nyuv2_test2.txt",
        gt_path="/mnt/nfs/workspace/syq/dataset/Eval/normal/nyuv2/",
        output_dir="result/nyuv2_normal/",
        dataset_arg="nyu",
    ),
    DatasetConfig(
        name="scannet",
        file_list="./data_split/scannet_normals/scannet_test.txt",
        gt_path="/mnt/nfs/workspace/syq/dataset/Eval/normal/scannet/",
        output_dir="result/scannet_normal/",
        dataset_arg="scannet",
    ),
    DatasetConfig(
        name="ibims",
        file_list="./data_split/ibims_normals/ibims_test2.txt",
        gt_path="/mnt/nfs/workspace/syq/dataset/Eval/normal/ibims/",
        output_dir="result/ibims_normal/",
        dataset_arg="ibims",
    ),
    DatasetConfig(
        name="diode",
        file_list="./data_split/diode_normals/diode_test.txt",
        gt_path="/mnt/nfs/workspace/syq/dataset/Eval/normal/diode/",
        output_dir="result/diode_normal/",
        dataset_arg="diode",
    ),
    # DatasetConfig(
    #     name="oasis",
    #     file_list="./data_split/oasis_normals/oasis_test2.txt",
    #     gt_path="/mnt/nfs/workspace/syq/dataset/Eval/normal/oasis/",
    #     output_dir="result/oasis_normal/",
    #     dataset_arg="oasis",
    # )
]

DATASETS_MATTING: List[DatasetConfig] = [
    # DatasetConfig(
    #     name="comp",
    #     file_list="./data_split/comp_matting/filenames_test.txt",
    #     gt_path="/mnt/nfs/workspace/syq/dataset/matting/composition-1k",
    #     output_dir="result/comp_matting/",
    #     dataset_arg="comp",
    # ),
    # DatasetConfig(
    #     name="p3m",
    #     file_list="./data_split/P3M_matting/filenames_val_P.txt",
    #     gt_path="/mnt/nfs/workspace/syq/dataset/matting/P3M-10k",
    #     output_dir="result/p3m_matting/",
    #     dataset_arg="p3m",
    # ),

    DatasetConfig(
        name="aim",
        file_list="./data_split/AIM_matting/filenames_val.txt",
        gt_path="/mnt/nfs/workspace/syq/dataset/matting/AIM-500",
        output_dir="result/aim_matting/",
        dataset_arg="aim",
    ),
    DatasetConfig(
        name="p3m-np",
        file_list="./data_split/P3M_matting/filenames_val_NP.txt",
        gt_path="/mnt/nfs/workspace/syq/dataset/matting/P3M-10k",
        output_dir="result/p3m_matting/",
        dataset_arg="p3m-np",
    ),
    DatasetConfig(
        name="am",
        file_list="./data_split/AM_matting/filenames_val.txt",
        gt_path="/mnt/nfs/workspace/syq/dataset/matting/AM-2k",
        output_dir="result/am_matting/",
        dataset_arg="am",
    )
]

def read_file_list(file_list_path: str, base_dir: str, extra_cols = 0) -> List[str]:
    """读取文件列表，返回拼接好的绝对路径列表。兼容每行只有一个 path 或带其它列的情况。"""
    files = []
    files_extra = [[] for _ in range(extra_cols)]
    base = Path(base_dir)
    with open(file_list_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            rel = parts[0]

            # 如果第二列是 "None"，跳过（你原来的逻辑）
            if len(parts) > 1 and parts[1] == "None":
                continue
            p = base / rel
            files.append(str(p))
            if extra_cols > 0:
                for i in range(extra_cols):
                    rel2 = parts[i+1] if len(parts) > i+1 else "None"
                    files_extra[i].append(str(base / rel2) if rel2 != "None" else "None")
    if extra_cols > 0:
        return files, files_extra
    return files


def create_batches(files: List[str], batch_size: int) -> List[List[str]]:
    """将文件列表分成批次"""
    batches = []
    for i in range(0, len(files), batch_size):
        batch = files[i:i + batch_size]
        batches.append(batch)
    return batches


def load_batch_images(file_batch: List[str], transform=None) -> List:
    """加载一个批次的图像"""
    images = []
    valid_files = []
    if transform is not None:
        for file in file_batch:
            try:
                img = transform(file)
                images.append(img)
                valid_files.append(file)
            except Exception as e:
                print(f"Warning: Failed to load {file}: {e}")
                # 继续处理其他文件
    else:
        for file in file_batch:
            try:
                img = np.array(Image.open(file))
                images.append(img)
                valid_files.append(file)
            except Exception as e:
                print(f"Warning: Failed to load {file}: {e}")
                # 继续处理其他文件
    
    return images, valid_files


def evaluate_dataset(pipe,
                     transform,
                     ds_cfg: DatasetConfig,
                     batch_size: int = 4,
                     max_samples: Optional[int] = None,
                     inference_kwargs: Optional[dict] = None,
                     cur_step: int = 3000,
                     args=None,):
    print(f"\n=== Evaluate dataset: {ds_cfg.name} (batch_size={batch_size}) ===")
    trimaps,alphas = None, None
    if args.task == "depth" or args.task == "normal":
        files = read_file_list(ds_cfg.file_list, ds_cfg.gt_path)
    elif args.task == "matting":
        files, extras = read_file_list(ds_cfg.file_list, ds_cfg.gt_path, extra_cols=2)
        trimaps, alphas = extras
    if max_samples is not None:
        files = files[:max_samples]
    print(f"Total {len(files)} files for eval (dataset={ds_cfg.name})")

    # ds_cfg.output_dir = os.path.join(ds_cfg.output_dir, f"test_step{cur_step}")
    out_base = Path(ds_cfg.output_dir)
    out_base.mkdir(parents=True, exist_ok=True)

    # 创建批次
    batches = create_batches(files, batch_size)
    if trimaps is not None:
        trimap_batches = create_batches(trimaps, batch_size)

    if alphas is not None:
        alpha_batches = create_batches(alphas, batch_size)
        print(f" using alphas: {len(alphas)}")
    failures = []
    total_processed = 0
    if trimaps is not None:
        # Matting: 生成多类型视觉提示 trimap/mask/bbox/points + 可选 coords
        for batch_files, alpha_files in tqdm(zip(batches, alpha_batches), desc=f"generating {ds_cfg.name}", total=len(batches),disable=True):
            batch_images, valid_files = load_batch_images(batch_files, transform)
            alpha_images, _ = load_batch_images(alpha_files)
            if not batch_images:
                failures.extend([(f, "Failed to load image") for f in batch_files])
                continue
            visual_prompts = []
            coords_list = []  # 每个样本坐标
            for img_tensor, alpha in zip(batch_images, alpha_images):
                if alpha.ndim == 3:
                    alpha = alpha[:,:,0]
                if alpha.dtype != np.float32:
                    alpha = alpha.astype(np.float32)/255.0
                alpha = np.clip(alpha, 0.0, 1.0)
                vp = None
                vp_coords = None
                if args.matting_prompt is not None:
                    if args.matting_prompt == "trimap":
                        vp, vp_coords = gen_trimap((alpha * 255).astype(np.uint8))
                        if vp is not None:
                            vp = (vp / 255.0).astype(np.float32)
                    elif args.matting_prompt == "mask":
                        vp, vp_coords = gen_mask(alpha)
                    elif args.matting_prompt == "bbox":
                        vp, vp_coords = gen_bbox(alpha, 0.01)
                    elif args.matting_prompt == "points":
                        vp, vp_coords = gen_points(alpha)

                    else:
                        raise ValueError(f"Unsupported matting_prompt {args.matting_prompt}")
                    # resize -> 768x768 if needed
                    if isinstance(vp, np.ndarray):
                        if vp.shape != (768, 768):
                            import cv2
                            vp = cv2.resize(vp, (768, 768), interpolation=cv2.INTER_LINEAR)
                        vp_tensor = torch.from_numpy(vp).unsqueeze(0).to(img_tensor.dtype)
                    else:
                        vp_tensor = vp
                        if vp_tensor.dim() == 2:
                            vp_tensor = vp_tensor.unsqueeze(0)
                    vp_tensor = (vp_tensor * 2 - 1).repeat(3, 1, 1)
                    visual_prompts.append(vp_tensor)
                    # if args.use_coor_input:
                    if vp_coords is not None:
                        coords_list.append(vp_coords.astype(np.float32))
                    else:
                        coords_list.append(np.array([0, 0, 1, 1], dtype=np.float32))
            # 组装推理参数
            if len(batch_images) == 1:
                if visual_prompts != []:
                    kontext_images = [batch_images[0], visual_prompts[0]]
                else:
                    kontext_images = batch_images[0]
                pipe_kwargs = dict(
                    prompt=f"Transform to {args.task} map while maintaining original composition",
                    kontext_images=kontext_images,
                    height=768, width=768,
                    embedded_guidance=inference_kwargs.get("embedded_guidance", 4),
                    num_inference_steps=inference_kwargs.get("num_inference_steps", 4),
                    seed=inference_kwargs.get("seed", 42),
                    output_type="np",
                    rand_device=inference_kwargs.get("rand_device", "cuda"),
                    task=args.task,
                )
                if args.use_coor_input and len(coords_list) > 0:
                    vpc_tensor = torch.from_numpy(coords_list[0]).to(torch.float32)
                    if vpc_tensor.dim() == 1:
                        vpc_tensor = vpc_tensor.unsqueeze(0).to("cuda")
                    pipe_kwargs["visual_prompt_coords"] = vpc_tensor
                out_np_batch = pipe(**pipe_kwargs)
                if not isinstance(out_np_batch, list):
                    out_np_batch = [out_np_batch]
            else:
                images_stack = torch.stack(batch_images).to("cuda")
                if visual_prompts != []:
                    prompts_stack = torch.stack(visual_prompts).to("cuda")
                    kontext_images = [images_stack,prompts_stack]
                else:
                    kontext_images = images_stack
                pipe_kwargs = dict(
                    prompt=[f"Transform to {args.task} map while maintaining original composition"] * len(batch_images),
                    kontext_images=kontext_images,
                    height=768, width=768,
                    embedded_guidance=inference_kwargs.get("embedded_guidance", 4),
                    num_inference_steps=inference_kwargs.get("num_inference_steps", 4),
                    seed=inference_kwargs.get("seed", 42),
                    output_type="np",
                    rand_device=inference_kwargs.get("rand_device", "cuda"),
                    task=args.task,
                    
                )
                if args.use_coor_input and len(coords_list) == len(batch_images):
                    vpc_tensor = torch.from_numpy(np.stack(coords_list, axis=0)).to(torch.float32).to("cuda")
                    pipe_kwargs["visual_prompt_coords"] = vpc_tensor
                out_np_batch = pipe(**pipe_kwargs)
            
            for file, out_np in zip(valid_files, out_np_batch):
                rel_out = Path(file).relative_to(Path(ds_cfg.gt_path))
                save_to = out_base / rel_out
                save_to = save_to.with_suffix(".npy")
                save_to.parent.mkdir(parents=True, exist_ok=True)
                np.save(str(save_to), out_np)
            # 额外保存点提示 visual_prompts
            # for i, file in enumerate(valid_files):
            #     if args.matting_prompt == "points":
            #         # 只保存坐标，每次用"a"的方式添加写入到txt文件中
            #         with open(out_base / "points_coords.txt", "a") as pf:
            #             rel_out = Path(file).relative_to(Path(ds_cfg.gt_path))
            #             coords = coords_list[valid_files.index(file)]
            #             coord_str = ",".join([str(c) for c in coords])
            #             pf.write(f"{rel_out.as_posix()}: {coord_str}\n")
            #     total_processed += 1
    else:
        for batch_files in tqdm(batches, desc=f"generating {ds_cfg.name}",disable=True):
            # 加载这个批次的图像
            batch_images, valid_files = load_batch_images(batch_files, transform)
            if not batch_images:
                # 如果整个批次都加载失败，跳过
                failures.extend([(f, "Failed to load image") for f in batch_files])
                continue
            
            # 批量推理
            # 注意：这里假设 pipe 支持批量输入，如果不支持需要修改
            if len(batch_images) == 1:
                # 单个图像的情况
                out_np_batch = pipe(
                    prompt=f"Transform to {args.task} map while maintaining original composition",
                    kontext_images=batch_images[0],
                    height=768, width=768,
                    embedded_guidance=inference_kwargs.get("embedded_guidance", 4),
                    num_inference_steps=inference_kwargs.get("num_inference_steps", 4),
                    seed=inference_kwargs.get("seed", 42),
                    output_type="np",
                    rand_device=inference_kwargs.get("rand_device", "cuda"),
                    task=args.task,
                    # deterministic_flow=True, ############ only for debug ####################
                )
                # 确保输出是列表形式
                if not isinstance(out_np_batch, list):
                    out_np_batch = [out_np_batch]
            else:
                # 批量推理 - 这里可能需要根据你的 pipeline 实现进行调整
                # 方案1: 如果 pipeline 支持批量输入
                out_np_batch = pipe(
                    prompt=[f"Transform to {args.task} map while maintaining original composition"] * len(batch_images),
                    kontext_images=torch.stack(batch_images),
                    height=768, width=768,
                    embedded_guidance=inference_kwargs.get("embedded_guidance", 4),
                    num_inference_steps=inference_kwargs.get("num_inference_steps", 4),
                    seed=inference_kwargs.get("seed", 42),
                    output_type="np",
                    rand_device='cuda',
                    task=args.task,
                )
            
            # 保存结果
            for file, out_np in zip(valid_files, out_np_batch):
                rel_out = Path(file).relative_to(Path(ds_cfg.gt_path))
                save_to = out_base / rel_out
                save_to = save_to.with_suffix(".npy")
                save_to.parent.mkdir(parents=True, exist_ok=True)
                np.save(str(save_to), out_np)
                total_processed += 1
                    
    
    print(f"Generation finished for {ds_cfg.name}. Processed: {total_processed}, Failures: {len(failures)}")
    if failures:
        # 简短输出若干错误示例
        print("Some failures (showing up to 5):")
        for f, err in failures[:5]:
            print(f" - {f}: {err}")

    # 调用评估脚本
    max_depth_eval = {"scannet": 10.0, "nyu": 10.0, "kitti": 80.0, "eth3d": 99999, "diode": 80.0}.get(ds_cfg.dataset_arg, 80.0)


    if args.task == "depth":
        eval_args = argparse.Namespace(
        pred_path=ds_cfg.output_dir,
        gt_path=ds_cfg.gt_path,
        dataset=ds_cfg.dataset_arg,
        eigen_crop=True if ds_cfg.dataset_arg == "kitti" else False,
        garg_crop=False,
        min_depth_eval=1e-3,
        max_depth_eval=max_depth_eval,
        do_kb_crop=True if ds_cfg.dataset_arg == "kitti" else False,
        no_verbose=False,
        using_log=("log" in args.state_dict),
        using_disp=("disp" in args.state_dict or "inverse" in args.state_dict),
        using_sqrt=("sqrt" in args.state_dict),
        using_sqrt_disp=("sqrt_disp" in args.state_dict),
        using_pdf=("pdf" in args.state_dict)
    )
        print(eval_args)
        try:
            eval_depth(eval_args)
        except Exception as e:
            print(f"Evaluation failed for {ds_cfg.name}: {e}")
    elif args.task == "normal":
        eval_args = argparse.Namespace(
            pred_path=ds_cfg.output_dir,
            gt_path=ds_cfg.gt_path,
            dataset=ds_cfg.dataset_arg,
        )
        print(eval_args)
        try:
            eval_normal(eval_args)
        except Exception as e:
            print(f"Evaluation failed for {ds_cfg.name}: {e}")
    elif args.task == "matting":
        eval_args = argparse.Namespace(
            pred_path=ds_cfg.output_dir,
            gt_path=ds_cfg.gt_path,
            dataset=ds_cfg.dataset_arg,
        )
        print(eval_args)
        try:
            eval_matting(eval_args)
        except Exception as e:
            print(f"Evaluation failed for {ds_cfg.name}: {e}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cur_step", type=int, default=5700, help="当前评估的训练步数，用于结果目录命名")
    parser.add_argument("--model_root", type=str, default="./FLUX.1-Kontext-dev", help="Flux model root directory")
    # parser.add_argument("--state_dict", type=str, default=f"models/train/kontext/bs64_mask/step-@@.safetensors", help="训练好的 state_dict path to load")
    # parser.add_argument("--state-dict", type=str, default=f"models/train/kontext/bs64_log_cons/step-{cur_step}.safetensors", help="训练好的 state_dict path to load")
    # parser.add_argument("--state_dict", type=str, default=f"models/train/kontext/bs64_sqrt_cons/step-@@.safetensors", help="训练好的 state_dict path to load")
    # parser.add_argument("--state-dict", type=str, default=f"models/train/kontext/bs64_sqrt_deter_zero/step-@@.safetensors", help="训练好的 state_dict path to load")
    parser.add_argument("--state_dict", type=str, default=f"models/train/kontext_normal/bs16_flux_cons/step-@@.safetensors", help="训练好的 state_dict path to load")
    # parser.add_argument("--state-dict", type=str, default=f"models/train/kontext_normal/bs16_cons/step-@@.safetensors", help="训练好的 state_dict path to load")
    # parser.add_argument("--state-dict", type=str, default=f"models/train/kontext_normal/bs16_deter_zero/step-@@.safetensors", help="训练好的 state_dict path to load")
    # parser.add_argument("--state-dict", type=str, default=f"models/train/kontext_matting/bs16_cons_mixSDMatte_points/step-@@.safetensors", help="训练好的 state_dict path to load")
    # parser.add_argument("--state-dict", type=str, default=f"models/train/kontext_matting/bs16_cons_mixSDMatte_empty/step-{cur_step}.safetensors", help="训练好的 state_dict path to load")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--torch_dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--datasets", type=str, default=None, help="逗号分隔的数据集名（使用 DATASETS 中定义的 name 字段）")
    parser.add_argument("--max_samples", type=int, default=None, help="便捷：只跑每个数据集的前 N 张做快速测试")
    parser.add_argument("--batch_size", type=int, default=1, help="批量推理的批次大小")
    parser.add_argument("--embedded_guidance", type=float, default=1, help="推理时的 embedded_guidance 强度")
    parser.add_argument("--task", type=str, default="depth", choices=["depth", "normal","matting"], help="评估任务类型，决定用哪个 DATASETS 列表")
    parser.add_argument("--matting_prompt", type=str, default=None, choices=["trimap", "mask", "bbox", "points"], help="(matting) 视觉提示类型")
    parser.add_argument("--use_coor_input", action="store_true", help="是否传入 visual_prompt_coords (bbox/points 等坐标 embedding)")
    parser.add_argument("--hw", type=str, default="768x768")
    parser.add_argument("--inference_steps", type=int, default=1, help="推理时的采样步数")
    args = parser.parse_args()
    # 选择要跑的 datasets
    # if "normal" in args.state_dict:
    #     args.task = "normal"
    # elif "matting" in args.state_dict:
    #     args.task = "matting"
    if args.task == "depth":
        DATASETS = DATASETS_DEPTH
    elif args.task == "normal":
        DATASETS = DATASETS_NORMAL
    elif args.task == "matting":
        DATASETS = DATASETS_MATTING
    if args.datasets:
        wanted = {n.strip() for n in args.datasets.split(",") if n.strip()}
        datasets = [d for d in DATASETS if d.name in wanted]
        print(f"Selected datasets: {[d.name for d in datasets]}")
        if not datasets:
            raise ValueError(f"No matching dataset in DATASETS for names: {wanted}")
    else:
        datasets = DATASETS
        print(f"No --datasets specified, will evaluate all: {[d.name for d in datasets]}")

    # load pipeline once
    torch_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.torch_dtype]
    print("Loading FluxImagePipeline ...")
    pipe = FluxImagePipeline.from_pretrained(
        torch_dtype=torch_dtype,
        device=args.device,
        model_configs=parse_flux_model_configs(args.model_root)
    )
    state_dict = load_state_dict(args.state_dict)
    pipe.dit.load_state_dict(state_dict)
    # transform reuse
    h, w = map(int, args.hw.split("x"))
    print(f"Eval with {h}x{w}")
    transform = UnifiedDataset.default_image_operator(height=h, width=w)

    inference_kwargs = dict(
        num_inference_steps=args.inference_steps,
        seed=42, 
        rand_device=args.device.split(":")[0] if ":" in args.device else args.device,
        embedded_guidance=args.embedded_guidance
    )

    for ds in datasets:
        evaluate_dataset(
            pipe, 
            transform, 
            ds, 
            batch_size=args.batch_size, 
            max_samples=args.max_samples,
            inference_kwargs=inference_kwargs,
            args=args,
        )

    print("\nAll done.")


if __name__ == "__main__":
    main()
