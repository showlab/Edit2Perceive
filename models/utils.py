import torch, os
from safetensors import safe_open
from contextlib import contextmanager
import hashlib

@contextmanager
def init_weights_on_device(device = torch.device("meta"), include_buffers :bool = False):
    
    old_register_parameter = torch.nn.Module.register_parameter
    if include_buffers:
        old_register_buffer = torch.nn.Module.register_buffer
    
    def register_empty_parameter(module, name, param):
        old_register_parameter(module, name, param)
        if param is not None:
            param_cls = type(module._parameters[name])
            kwargs = module._parameters[name].__dict__
            kwargs["requires_grad"] = param.requires_grad
            module._parameters[name] = param_cls(module._parameters[name].to(device), **kwargs)

    def register_empty_buffer(module, name, buffer, persistent=True):
        old_register_buffer(module, name, buffer, persistent=persistent)
        if buffer is not None:
            module._buffers[name] = module._buffers[name].to(device)
            
    def patch_tensor_constructor(fn):
        def wrapper(*args, **kwargs):
            kwargs["device"] = device
            return fn(*args, **kwargs)

        return wrapper
    
    if include_buffers:
        tensor_constructors_to_patch = {
            torch_function_name: getattr(torch, torch_function_name)
            for torch_function_name in ["empty", "zeros", "ones", "full"]
        }
    else:
        tensor_constructors_to_patch = {}
    
    try:
        torch.nn.Module.register_parameter = register_empty_parameter
        if include_buffers:
            torch.nn.Module.register_buffer = register_empty_buffer
        for torch_function_name in tensor_constructors_to_patch.keys():
            setattr(torch, torch_function_name, patch_tensor_constructor(getattr(torch, torch_function_name)))
        yield
    finally:
        torch.nn.Module.register_parameter = old_register_parameter
        if include_buffers:
            torch.nn.Module.register_buffer = old_register_buffer
        for torch_function_name, old_torch_function in tensor_constructors_to_patch.items():
            setattr(torch, torch_function_name, old_torch_function)

def load_state_dict_from_folder(file_path, torch_dtype=None):
    state_dict = {}
    for file_name in sorted(os.listdir(file_path)):
        if "." in file_name and file_name.split(".")[-1] in [
            "safetensors", "bin", "ckpt", "pth", "pt"
        ]:
            state_dict.update(load_state_dict(os.path.join(file_path, file_name), torch_dtype=torch_dtype))
    return state_dict


def load_state_dict(file_path, torch_dtype=None, device="cpu"):
    if file_path.endswith(".safetensors"):
        return load_state_dict_from_safetensors(file_path, torch_dtype=torch_dtype, device=device)
    else:
        return load_state_dict_from_bin(file_path, torch_dtype=torch_dtype, device=device)


def load_state_dict_from_safetensors(file_path, torch_dtype=None, device="cpu"):
    state_dict = {}
    with safe_open(file_path, framework="pt", device=str(device)) as f:
        for k in f.keys():
            state_dict[k] = f.get_tensor(k)
            if torch_dtype is not None:
                state_dict[k] = state_dict[k].to(torch_dtype)
    return state_dict


def load_state_dict_from_bin(file_path, torch_dtype=None, device="cpu"):
    state_dict = torch.load(file_path, map_location=device, weights_only=True)
    if torch_dtype is not None:
        for i in state_dict:
            if isinstance(state_dict[i], torch.Tensor):
                state_dict[i] = state_dict[i].to(torch_dtype)
    return state_dict


def search_for_embeddings(state_dict):
    embeddings = []
    for k in state_dict:
        if isinstance(state_dict[k], torch.Tensor):
            embeddings.append(state_dict[k])
        elif isinstance(state_dict[k], dict):
            embeddings += search_for_embeddings(state_dict[k])
    return embeddings


def search_parameter(param, state_dict):
    for name, param_ in state_dict.items():
        if param.numel() == param_.numel():
            if param.shape == param_.shape:
                if torch.dist(param, param_) < 1e-3:
                    return name
            else:
                if torch.dist(param.flatten(), param_.flatten()) < 1e-3:
                    return name
    return None


def build_rename_dict(source_state_dict, target_state_dict, split_qkv=False):
    matched_keys = set()
    with torch.no_grad():
        for name in source_state_dict:
            rename = search_parameter(source_state_dict[name], target_state_dict)
            if rename is not None:
                print(f'"{name}": "{rename}",')
                matched_keys.add(rename)
            elif split_qkv and len(source_state_dict[name].shape)>=1 and source_state_dict[name].shape[0]%3==0:
                length = source_state_dict[name].shape[0] // 3
                rename = []
                for i in range(3):
                    rename.append(search_parameter(source_state_dict[name][i*length: i*length+length], target_state_dict))
                if None not in rename:
                    print(f'"{name}": {rename},')
                    for rename_ in rename:
                        matched_keys.add(rename_)
    for name in target_state_dict:
        if name not in matched_keys:
            print("Cannot find", name, target_state_dict[name].shape)


def search_for_files(folder, extensions):
    files = []
    if os.path.isdir(folder):
        for file in sorted(os.listdir(folder)):
            files += search_for_files(os.path.join(folder, file), extensions)
    elif os.path.isfile(folder):
        for extension in extensions:
            if folder.endswith(extension):
                files.append(folder)
                break
    return files


def convert_state_dict_keys_to_single_str(state_dict, with_shape=True):
    keys = []
    for key, value in state_dict.items():
        if isinstance(key, str):
            if isinstance(value, torch.Tensor):
                if with_shape:
                    shape = "_".join(map(str, list(value.shape)))
                    keys.append(key + ":" + shape)
                keys.append(key)
            elif isinstance(value, dict):
                keys.append(key + "|" + convert_state_dict_keys_to_single_str(value, with_shape=with_shape))
    keys.sort()
    keys_str = ",".join(keys)
    return keys_str


def split_state_dict_with_prefix(state_dict):
    keys = sorted([key for key in state_dict if isinstance(key, str)])
    prefix_dict = {}
    for key in  keys:
        prefix = key if "." not in key else key.split(".")[0]
        if prefix not in prefix_dict:
            prefix_dict[prefix] = []
        prefix_dict[prefix].append(key)
    state_dicts = []
    for prefix, keys in prefix_dict.items():
        sub_state_dict = {key: state_dict[key] for key in keys}
        state_dicts.append(sub_state_dict)
    return state_dicts


def hash_state_dict_keys(state_dict, with_shape=True):
    keys_str = convert_state_dict_keys_to_single_str(state_dict, with_shape=with_shape)
    keys_str = keys_str.encode(encoding="UTF-8")
    return hashlib.md5(keys_str).hexdigest()


import imageio, os, torch, warnings, torchvision, argparse, json
from utils import ModelConfig
from models.utils import load_state_dict
from peft import LoraConfig, inject_adapter_in_model
from PIL import Image
import pandas as pd
from tqdm import tqdm
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
import numpy as np
import cv2
from bitsandbytes.optim import AdamW8bit
from models.unified_dataset import gen_mask, gen_bbox, gen_points
from utils.eval_depth import test as eval_depth
from utils.eval_normal import test as eval_normal
from utils.eval_matting import test as eval_matting
from lora.flux_lora import FluxLoRALoader

# 找到最后的一个检查点
def find_latest_checkpoint(folder):
    if not os.path.exists(folder):
        return None
    checkpoint_files = [f for f in os.listdir(folder) if f.startswith("step-") and f.endswith(".safetensors")]
    if not checkpoint_files:
        return None
    # 提取步数并找到最大的那个
    latest_checkpoint = max(checkpoint_files, key=lambda x: int(x.split('-')[1].split('.')[0]))
    return os.path.join(folder, latest_checkpoint)

class DiffusionTrainingModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
        
    def to(self, *args, **kwargs):
        for name, model in self.named_children():
            model.to(*args, **kwargs)
        return self
        
        
    def trainable_modules(self):
        trainable_modules = filter(lambda p: p.requires_grad, self.parameters())
        return trainable_modules
    
    
    def trainable_param_names(self):
        trainable_param_names = list(filter(lambda named_param: named_param[1].requires_grad, self.named_parameters()))
        trainable_param_names = set([named_param[0] for named_param in trainable_param_names])
        return trainable_param_names
    
    
    def add_lora_to_model(self, model, target_modules, lora_rank, lora_alpha=None, upcast_dtype=None):
        if lora_alpha is None:
            lora_alpha = lora_rank
        lora_config = LoraConfig(r=lora_rank, lora_alpha=lora_alpha, target_modules=target_modules)
        model = inject_adapter_in_model(lora_config, model)
        if upcast_dtype is not None:
            for param in model.parameters():
                if param.requires_grad:
                    param.data = param.to(upcast_dtype)
        return model


    def mapping_lora_state_dict(self, state_dict):
        new_state_dict = {}
        for key, value in state_dict.items():
            if "lora_A.weight" in key or "lora_B.weight" in key:
                new_key = key.replace("lora_A.weight", "lora_A.default.weight").replace("lora_B.weight", "lora_B.default.weight")
                new_state_dict[new_key] = value
            elif "lora_A.default.weight" in key or "lora_B.default.weight" in key:
                new_state_dict[key] = value
        return new_state_dict


    def export_trainable_state_dict(self, state_dict, remove_prefix=None):
        trainable_param_names = self.trainable_param_names()
        state_dict = {name: param for name, param in state_dict.items() if name in trainable_param_names}
        if remove_prefix is not None:
            state_dict_ = {}
            for name, param in state_dict.items():
                if name.startswith(remove_prefix):
                    name = name[len(remove_prefix):]
                state_dict_[name] = param
            state_dict = state_dict_
        return state_dict
    
    
    def transfer_data_to_device(self, data, device):
        for key in data:
            if isinstance(data[key], torch.Tensor):
                data[key] = data[key].to(device)
        return data
    
    
    def parse_model_configs(self, model_paths, model_id_with_origin_paths, enable_fp8_training=False):
        offload_dtype = torch.float8_e4m3fn if enable_fp8_training else None
        model_configs = []
        if model_paths is not None:
            # model_paths = json.loads(model_paths)
            model_paths = model_paths.split(",")
            model_configs += [ModelConfig(path=path, offload_dtype=offload_dtype) for path in model_paths]
        if model_id_with_origin_paths is not None:
            model_id_with_origin_paths = model_id_with_origin_paths.split(",")
            model_configs += [ModelConfig(model_id=i.split(":")[0], origin_file_pattern=i.split(":")[1], offload_dtype=offload_dtype) for i in model_id_with_origin_paths]
        return model_configs


    def switch_pipe_to_training_mode(
        self,
        pipe,
        trainable_models,
        lora_base_model=None, lora_target_modules=None, lora_rank=None, lora_checkpoint=None,
        enable_fp8_training=False,
    ):
        # Scheduler
        pipe.scheduler.set_timesteps(1000, training=True)
        
        # Freeze untrainable models
        pipe.freeze_except([] if trainable_models is None else trainable_models.split(","))
        
        # Enable FP8 if pipeline supports
        if enable_fp8_training and hasattr(pipe, "_enable_fp8_lora_training"):
            pipe._enable_fp8_lora_training(torch.float8_e4m3fn)
        
        # Add LoRA to the base models
        if lora_base_model is not None:
            model = self.add_lora_to_model(
                getattr(pipe, lora_base_model),
                target_modules=lora_target_modules.split(","),
                lora_rank=lora_rank,
                upcast_dtype=pipe.torch_dtype,
            )
            if lora_checkpoint is not None:
                state_dict = load_state_dict(lora_checkpoint)
                loader = FluxLoRALoader(torch_dtype=torch.bfloat16, device="cuda")
                state_dict = loader.convert_state_dict(state_dict)
                state_dict = self.mapping_lora_state_dict(state_dict)
                load_result = model.load_state_dict(state_dict, strict=False)
                print(f"LoRA checkpoint loaded: {lora_checkpoint}, total {len(state_dict)} keys")
                if len(load_result[1]) > 0:
                    print(f"Warning, LoRA key mismatch! Unexpected keys in LoRA checkpoint: {load_result[1]}")
            setattr(pipe, lora_base_model, model)


class ModelLogger:
    def __init__(self, output_path, remove_prefix_in_ckpt=None, state_dict_converter=lambda x:x,
                 args=None):
        self.output_path = output_path
        self.remove_prefix_in_ckpt = remove_prefix_in_ckpt
        self.state_dict_converter = state_dict_converter
        self.num_steps = args.resume_steps
        # Evaluation related
        self.eval_steps = args.eval_steps
        self.eval_file_list = args.eval_file_list
        self.eval_output_dir = os.path.join(self.output_path, "eval")
        self.eval_prompt = args.default_caption
        self.eval_num_inference_steps = args.eval_num_inference_steps
        self.eval_embedded_guidance = args.eval_embedded_guidance
        self.eval_height = args.height
        self.eval_width = args.width
        self.trainable_models = args.trainable_models
        self.using_log = args.using_log
        self.using_disp = args.using_disp
        self.using_sqrt = args.using_sqrt
        self.using_sqrt_disp = args.using_sqrt_disp
        self.task = args.task
        self.deterministic_flow = args.deterministic_flow
        self.eval_args = argparse.Namespace(
            pred_path="result/nyu_depth_v2",  # 对应 --pred_path
            gt_path="../dataset/Eval/depth/nyuv2",  # 对应 --gt_path
            dataset="nyu",  # 目标脚本默认值（命令行没传，用默认）
            eigen_crop=False,  
            garg_crop=False,  
            min_depth_eval=1e-3, 
            max_depth_eval=10.0, 
            do_kb_crop=False,
            no_verbose=False,
            using_log=args.using_log,
            using_disp=args.using_disp,
            using_sqrt=args.using_sqrt,
            using_sqrt_disp=args.using_sqrt_disp,
            using_pdf=args.using_pdf,
        )
        self.matting_prompt = args.matting_prompt
        # lora related
        self.lora_base_model = args.lora_base_model
        self.lora_target_modules = args.lora_target_modules
        self.lora_rank = args.lora_rank
        self.lora_checkpoint = args.lora_checkpoint

    def on_step_end(self, accelerator, model, save_steps=None):
        self.num_steps += 1
        if save_steps is not None and self.num_steps % save_steps == 0:
            self.save_model(accelerator, model, f"step-{self.num_steps}.safetensors")


    def run_evaluation(self, model, full_size=False):
        if self.eval_steps is None:
            return
        if not full_size:
            files = self.eval_file_list[:10]
            output_dir = self.eval_output_dir
        else:
            files = self.eval_file_list[:654]
            output_dir = self.eval_output_dir + "_full"
        if len(files) == 0:
            return

        try:
            from models.unified_dataset import UnifiedDataset
            transform = UnifiedDataset.default_image_operator()
        except Exception as e:
            print(f"[ModelLogger][Eval] Failed to build transform: {e}")
            return
        pipe = getattr(model, 'pipe', None)
        if pipe is None:
            print("[ModelLogger][Eval] Model has no 'pipe' attribute; skipping evaluation.")
            return
        # Maintain relative folder structure
        print(f"[ModelLogger][Eval] Running evaluation on {len(files)} files (steps={self.num_steps}). Saving to {output_dir}")
        if self.task == "depth" or self.task == "normal":
            base_dir = f"/mnt/nfs/workspace/syq/dataset/Eval/{self.task}/nyuv2"
        elif self.task == "matting":
            base_dir = "/mnt/nfs/workspace/syq/dataset/matting/P3M-10k"
        else:
            raise ValueError(f"Unknown task {self.task}")
        os.makedirs(output_dir, exist_ok=True)
        for file in files:
            try:
                save_to = file.replace(base_dir, output_dir).replace(".png", ".npy").replace(".jpg", ".npy")
                os.makedirs(os.path.dirname(save_to), exist_ok=True)
                with torch.no_grad():
                    if "kontext" in output_dir:
                        if self.task == "depth" or self.task == "normal":
                            output = pipe(
                                prompt=self.eval_prompt,
                                kontext_images=transform(file),
                                height=self.eval_height,
                                width=self.eval_width,
                                embedded_guidance=self.eval_embedded_guidance,
                                num_inference_steps=self.eval_num_inference_steps,
                                seed=42,
                                output_type="np",
                                rand_device=pipe.device,  # ensure format
                                deterministic_flow=self.deterministic_flow,
                                task=self.task,
                            )
                        elif self.task == "matting":
                            alpha = np.array(Image.open(file.replace("blurred_image","mask").replace("original_image","mask").replace(".jpg",".png")).convert("L")) / 255.0
                            if self.matting_prompt is not None:
                                if self.matting_prompt == "trimap":
                                    visual_prompt = transform(file.replace("blurred_image","trimap").replace("original_image","trimap").replace(".jpg",".png"))
                                elif self.matting_prompt == "mask":
                                    visual_prompt,visual_prompt_coords = gen_mask(alpha)
                                elif self.matting_prompt == "bbox":
                                    visual_prompt,visual_prompt_coords = gen_bbox(alpha,0)
                                elif self.matting_prompt == "points":
                                    visual_prompt,visual_prompt_coords = gen_points(alpha,radius=30)
                                
                                if isinstance(visual_prompt, np.ndarray):
                                    # resize to 1/8 size
                                    # visual_prompt = cv2.resize(visual_prompt, (self.eval_width // 8, self.eval_height // 8), interpolation=cv2.INTER_NEAREST)
                                    # visual_prompt = torch.from_numpy(visual_prompt).unsqueeze(0).to(torch.bfloat16).to("cuda")
                                    # visual_prompt_coords = torch.from_numpy(visual_prompt_coords).to(torch.bfloat16).to("cuda")
                                    # del alpha
                                    visual_prompt = cv2.resize(visual_prompt,(self.eval_width,self.eval_height),interpolation=cv2.INTER_LINEAR)
                                    visual_prompt = torch.from_numpy(visual_prompt*2 - 1).repeat(3,1,1).to(pipe.device).to(torch.bfloat16)
                                kontext_images = [transform(file),visual_prompt]
                            output = pipe(
                                prompt=self.eval_prompt,
                                kontext_images=kontext_images,
                                height=self.eval_height,
                                width=self.eval_width,
                                cfg_scale=self.eval_embedded_guidance,
                                num_inference_steps=self.eval_num_inference_steps,
                                seed=42,
                                output_type="np",
                                rand_device=pipe.device,  # ensure format
                                deterministic_flow=self.deterministic_flow,
                                task=self.task,
                                # kontext_masks=visual_prompt if self.matting_prompt is not None else None,
                            )
                    elif "qwen" in output_dir:
                        output = pipe(
                            prompt=self.eval_prompt,
                            edit_image=transform(file),
                            height=self.eval_height,
                            width=self.eval_width,
                            cfg_scale=self.eval_embedded_guidance,
                            num_inference_steps=self.eval_num_inference_steps,
                            seed=42,
                            output_type="np",
                            rand_device=pipe.device,  # ensure format
                            # deterministic_flow=self.deterministic_flow,
                            task=self.task,
                        )
                np.save(save_to, output)
            except Exception as e:
                print(f"[ModelLogger][Eval] Failed on file {file}: {e}")
        print("[ModelLogger][Eval] Inference finished.")
        self.eval_args.pred_path = output_dir
        if self.task == "depth":
            return eval_depth(self.eval_args)
        elif self.task == "normal":
            self.eval_args.gt_path = "/mnt/nfs/workspace/syq/dataset/Eval/normal/nyuv2"
            return eval_normal(self.eval_args)
        elif self.task == "matting":
            self.eval_args.gt_path = "/mnt/nfs/workspace/syq/dataset/matting/P3M-10k"
            self.eval_args.dataset = "p3m-np"
            return eval_matting(self.eval_args)
        else:
            print(f"[ModelLogger][Eval] Unknown task {self.task}; skipping evaluation.")
            return

    def on_eval_step(self, accelerator, model: DiffusionTrainingModule):
        if self.eval_steps is None or self.num_steps % self.eval_steps != 0:
            return
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            results = self.run_evaluation(model)
            with open(os.path.join(self.eval_output_dir, "log.txt"), "a") as f:
                f.write(f"Step {self.num_steps} evaluation results:\n")
                f.write(results+"\n")
        accelerator.wait_for_everyone()
        
        model.switch_pipe_to_training_mode(model.pipe,self.trainable_models,None,None,None,None)


    def on_epoch_end(self, accelerator, model, epoch_id):
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            state_dict = accelerator.get_state_dict(model)
            state_dict = accelerator.unwrap_model(model).export_trainable_state_dict(state_dict, remove_prefix=self.remove_prefix_in_ckpt)
            state_dict = self.state_dict_converter(state_dict)
            os.makedirs(self.output_path, exist_ok=True)
            path = os.path.join(self.output_path, f"epoch-{epoch_id}.safetensors")
            accelerator.save(state_dict, path, safe_serialization=True)


    def on_training_end(self, accelerator, model, save_steps=None):
        if save_steps is not None and self.num_steps % save_steps != 0:
            self.save_model(accelerator, model, f"step-{self.num_steps}.safetensors")


    def save_model(self, accelerator, model, file_name):
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            state_dict = accelerator.get_state_dict(model)
            state_dict = accelerator.unwrap_model(model).export_trainable_state_dict(state_dict, remove_prefix=self.remove_prefix_in_ckpt)
            state_dict = self.state_dict_converter(state_dict)
            os.makedirs(self.output_path, exist_ok=True)
            path = os.path.join(self.output_path, file_name)
            accelerator.save(state_dict, path, safe_serialization=True)
            results = self.run_evaluation(model,full_size=True)
            with open(os.path.join(self.eval_output_dir+"_full", "log_full.txt"), "a") as f:
                f.write(f"Step {self.num_steps} evaluation results:\n")
                f.write(results+"\n")
        accelerator.wait_for_everyone()

        model.switch_pipe_to_training_mode(model.pipe,self.trainable_models,None,None,None,None)
def launch_training_task(
    dataset: torch.utils.data.Dataset,
    model: DiffusionTrainingModule,
    model_logger: ModelLogger,
    dataset_sampler: torch.utils.data.Sampler = None,
    learning_rate: float = 1e-5,
    weight_decay: float = 1e-2,
    num_workers: int = 8,
    save_steps: int = None,
    num_epochs: int = 1,
    gradient_accumulation_steps: int = 1,
    find_unused_parameters: bool = False,
    args = None,
):
    if args is not None:
        learning_rate = args.learning_rate
        weight_decay = args.weight_decay
        num_workers = args.dataset_num_workers
        save_steps = args.save_steps
        num_epochs = args.num_epochs
        gradient_accumulation_steps = args.gradient_accumulation_steps
        find_unused_parameters = args.find_unused_parameters
    
    if args.adamw8bit:
        print("Using 8-bit AdamW optimizer.")
        optimizer = AdamW8bit(model.trainable_modules(), lr=learning_rate, weight_decay=weight_decay)
    else:
        print("Using regular AdamW optimizer.") 
        optimizer = torch.optim.AdamW(model.trainable_modules(), lr=learning_rate, weight_decay=weight_decay)

    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)
    # dataloader = torch.utils.data.DataLoader(dataset, shuffle=True, collate_fn=lambda x: x[0], num_workers=num_workers)
    if dataset_sampler is None:
        # random batch sampler with batch_size = args.batch_size
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=num_workers)
    else:
        dataloader = torch.utils.data.DataLoader(dataset, batch_sampler=dataset_sampler, num_workers=num_workers)
    

    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=find_unused_parameters)],
    )
    if getattr(accelerator.state, "deepspeed_plugin", None) is not None:
        print("Using DeepSpeed with batch size on per GPU", args.batch_size)
        accelerator.state.deepspeed_plugin.deepspeed_config["train_micro_batch_size_per_gpu"] = int(args.batch_size)
    model, optimizer, dataloader, scheduler = accelerator.prepare(model, optimizer, dataloader, scheduler)
    
    for epoch_id in range(num_epochs):
        loss = None
        pbar = tqdm(dataloader, desc=f"Epoch {epoch_id}")
        for i,data in enumerate(pbar):
            with accelerator.accumulate(model):
                optimizer.zero_grad()
                # if dataset.load_from_cache:
                #     loss = model({}, inputs=data)
                # else:
                    # loss = model(data)
                loss = model(data)
                # if i%50==0 and accelerator.is_main_process:
                #     with open("debug_loss_uni_flux.txt","a") as f:
                #         f.write(f"step {i}: {loss.item()}\n")
                if isinstance(loss, tuple) or isinstance(loss, list):
                    if epoch_id >= args.extra_loss_start_epoch:
                        loss, extra_loss = loss[0], loss[1]
                        coeff_scale = loss.detach().cpu().item() / (extra_loss.detach().cpu().item()+1e-3)
                        coeff_step = max(0,epoch_id-args.extra_loss_start_epoch+i/len(dataloader))
                        loss = loss + extra_loss * coeff_scale * coeff_step
                    else:
                        loss = loss[0]
                pbar.set_description(f"Loss {loss.item():.4f}")
                accelerator.backward(loss)
                optimizer.step()
                torch.cuda.empty_cache()
                model_logger.on_step_end(accelerator, model, save_steps)
                # Evaluation hook
                model_logger.on_eval_step(accelerator, model)
                scheduler.step()
        if save_steps is None:
            model_logger.on_epoch_end(accelerator, model, epoch_id)
    model_logger.on_training_end(accelerator, model, save_steps)

def parse_flux_model_configs(root_path):
    # given the root path, and then load the following: 
    # text_encoder, text_encoder_2, tokenizer, tokenizer_2, transformer(a folder of 3 parts) / or flux1-kontext-dev.safetensors, vae
    model_configs = []
    _targets = ["flux1-kontext-dev.safetensors", "text_encoder/model.safetensors", "text_encoder_2","ae.safetensors"]
    if "kontext" not in root_path.lower():
        _targets = ["flux1-dev.safetensors", "text_encoder/model.safetensors", "text_encoder_2","ae.safetensors"]
    for model_name in _targets:
        model_path = os.path.join(root_path, model_name)
        model_configs.append(ModelConfig(path=model_path))
    return model_configs


def flux_parser():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument("--dataset_base_path", type=str, default="", required=True, help="Base path of the dataset.")
    parser.add_argument("--dataset_metadata_path", type=str, default=None, help="Path to the metadata file of the dataset.")
    parser.add_argument("--max_pixels", type=int, default=1024*1024, help="Maximum number of pixels per frame, used for dynamic resolution..")
    parser.add_argument("--height", type=int, default=768, help="Height of images. Leave `height` and `width` empty to enable dynamic resolution.")
    parser.add_argument("--width", type=int, default=768, help="Width of images. Leave `height` and `width` empty to enable dynamic resolution.")
    parser.add_argument("--data_file_keys", type=str, default="image", help="Data file keys in the metadata. Comma-separated.")
    parser.add_argument("--dataset_repeat", type=int, default=1, help="Number of times to repeat the dataset per epoch.")
    parser.add_argument("--model_paths", type=str, default=None, help="Paths to load models. In JSON format.")
    parser.add_argument("--model_id_with_origin_paths", type=str, default=None, help="Model ID with origin paths, e.g., Wan-AI/Wan2.1-T2V-1.3B:diffusion_pytorch_model*.safetensors. Comma-separated.")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--num_epochs", type=int, default=1, help="Number of epochs.")
    parser.add_argument("--output_path", type=str, default="./models", help="Output save path.")
    parser.add_argument("--remove_prefix_in_ckpt", type=str, default="pipe.dit.", help="Remove prefix in ckpt.")
    parser.add_argument("--trainable_models", type=str, default=None, help="Models to train, e.g., dit, vae, text_encoder.")
    parser.add_argument("--lora_base_model", type=str, default=None, help="Which model LoRA is added to.")
    parser.add_argument("--lora_target_modules", type=str, default="q,k,v,o,ffn.0,ffn.2", help="Which layers LoRA is added to.")
    parser.add_argument("--lora_rank", type=int, default=32, help="Rank of LoRA.")
    parser.add_argument("--lora_checkpoint", type=str, default=None, help="Path to the LoRA checkpoint. If provided, LoRA will be loaded from this checkpoint.")
    parser.add_argument("--extra_inputs", default=None, help="Additional model inputs, comma-separated.")
    parser.add_argument("--align_to_opensource_format", default=False, action="store_true", help="Whether to align the lora format to opensource format. Only for DiT's LoRA.")
    parser.add_argument("--use_gradient_checkpointing", default=False, action="store_true", help="Whether to use gradient checkpointing.")
    parser.add_argument("--use_gradient_checkpointing_offload", default=False, action="store_true", help="Whether to offload gradient checkpointing to CPU memory.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps.")
    parser.add_argument("--find_unused_parameters", default=False, action="store_true", help="Whether to find unused parameters in DDP.")
    parser.add_argument("--save_steps", type=int, default=None, help="Number of checkpoint saving invervals. If None, checkpoints will be saved every epoch.")
    parser.add_argument("--dataset_num_workers", type=int, default=16, help="Number of workers for data loading.")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay.")
    parser.add_argument("--mixed_sampler", default=False, action="store_true", help="Whether to use mixed sampler (for mixed models, e.g., Wan-AI/Wan2.2-I2V-A14B).")
    # normalization
    parser.add_argument("--depth_normalization",type=str, default=None, help="Normalization method for depth map")
    parser.add_argument("--using_log", default=False, action="store_true", help="Whether to use log for depth preprocessing.")
    parser.add_argument("--using_disp", default=False, action="store_true", help="Whether to use disp for depth preprocessing.")
    parser.add_argument("--using_sqrt",default=False, action="store_true", help="Whether to use sqrt for depth preprocessing.")
    parser.add_argument("--using_sqrt_disp",default=False, action="store_true", help="Whether to use sqrt for depth preprocessing.")
    parser.add_argument("--using_pdf", default=False, action="store_true", help="Whether to use pdf for depth preprocessing.")
    # text and visual prompt
    parser.add_argument("--with_mask", default=False, action="store_true", help="Whether to use mask for loss calculation, espcially for normal estimation.")
    parser.add_argument("--default_caption", type=str, default=None, help="Default caption for all training samples.")
    parser.add_argument("--matting_prompt", type=str, default=None,choices=["trimap", "mask", "bbox", "points"], help="Prompt for image matting.")
    parser.add_argument("--use_coor_input", default=False, action="store_true", help="Whether to use coordinate as model input.")
    parser.add_argument("--use_camera_intrinsics", default=False, action="store_true", help="Whether to use camera intrinsics as model input.")
    # Evaluation related arguments
    parser.add_argument("--eval_steps", type=int, default=5, help="Run evaluation every N training steps.")
    parser.add_argument("--eval_num_inference_steps", type=int, default=1, help="Number of inference steps for evaluation.")
    parser.add_argument("--eval_embedded_guidance", type=float, default=1, help="Embedded guidance for evaluation generation.")
    parser.add_argument("--eval_file_list", type=str, default="", help="A text file containing a list of file paths to be used for evaluation. If empty, no evaluation is performed.")
    # deterministic training
    parser.add_argument("--deterministic_flow", default=False, action="store_true", help="Whether to use deterministic flow for training.")
    # training
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for training.")
    parser.add_argument("--adamw8bit", default=False, action="store_true", help="Whether to use 8-bit Adam optimizer.")
    parser.add_argument("--multi_res_noise", default=False, action="store_true", help="Whether to use multi-resolution noise for training.")
    parser.add_argument("--resume", default=False, action="store_true", help="Whether to resume training from a checkpoint.")
    parser.add_argument("--task", type=str, default="depth", required=False, help="Task type, e.g., depth, normal.")
    # loss
    parser.add_argument("--extra_loss", type=str, default=None, help="Loss type for depth estimation, e.g., l1, l2, berhu.")
    parser.add_argument("--extra_loss_start_epoch", type=int, default=0, help="The start epoch for applying extra loss. before this epoch, only the main loss is used.")
    return parser