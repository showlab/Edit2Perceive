import torch, warnings, glob, os, types, math
import torch.nn.functional as F
import numpy as np
from PIL import Image
from einops import repeat, reduce
from typing import Optional, Union
from dataclasses import dataclass
from einops import rearrange

from tqdm import tqdm

from typing_extensions import Literal

from models.flowmatch_scheduler import FlowMatchScheduler
from prompters.flux_prompter import FluxPrompter
from models import ModelManager, load_state_dict, SD3TextEncoder1, FluxTextEncoder2, FluxDiT, FluxVAEEncoder, FluxVAEDecoder
from models.tiler import FastTileWorker
from utils import BasePipeline, ModelConfig, PipelineUnitRunner, PipelineUnit
from lora.flux_lora import FluxLoRALoader, FluxLoraPatcher, FluxLoRAFuser

from models.flux_dit import RMSNorm
from vram_management import gradient_checkpoint_forward, enable_vram_management, AutoWrappedModule, AutoWrappedLinear
from utils.cycle_loss import get_cycle_consistency_normal_loss,get_cycle_consistency_depth_loss,get_cycle_consistency_matting_loss,get_disperse_loss,ScaleAndShiftInvariantLoss
import matplotlib.pyplot as plt




def visualize_batch(batch, max_items=4, figsize=(12, 12)):
    """
    批量可视化函数
    batch: dict, 包含 'kontext_images', 'image', 'mask'
    max_items: 最多展示多少个样本
    """

    def prepare_image(tensor):
        """把tensor处理成matplotlib能显示的格式 (H,W,C) or (H,W)"""
        if torch.is_tensor(tensor):
            tensor = tensor.detach().cpu()
        if tensor.ndim == 3:  # (C,H,W)
            tensor = torch.clip(tensor, -1, 1)
            tensor = (tensor + 1) / 2  # -1~1 -> 0~1
            tensor = tensor.clamp(0, 1)
            tensor = tensor.permute(1, 2, 0)  # (H,W,C)
        elif tensor.ndim == 2:  # (H,W) mask
            tensor = tensor
        return tensor.numpy()
    if isinstance(batch["kontext_images"], list):
        # kontext_images is a list of 2 tensors:
        kontexts = [img.to(torch.float32) for img in batch["kontext_images"]]
        images   = batch["input_image"].to(torch.float32)
        masks    = batch["mask"].to(torch.float32)

        B = images.shape[0]
        n_show = min(B, max_items)
        fig, axs = plt.subplots(n_show, 4, figsize=(figsize[0], figsize[1] * n_show / 4))
        if n_show == 1:
            axs = [axs]  # 保持二维结构
        for i in range(n_show):
            kontext1 = prepare_image(kontexts[0][i])
            kontext2 = prepare_image(kontexts[1][i])
            image    = prepare_image(images[i])
            mask     = prepare_image(masks[i])
            axs[i][0].imshow(kontext1)
            axs[i][0].set_title(f"kontext_images[0][{i}]")
            axs[i][0].axis("off")

            axs[i][1].imshow(kontext2)
            axs[i][1].set_title(f"kontext_images[1][{i}]")
            axs[i][1].axis("off")

            axs[i][2].imshow(image)
            axs[i][2].set_title(f"image[{i}]")
            axs[i][2].axis("off")

            axs[i][3].imshow(mask, cmap="gray")
            axs[i][3].set_title(f"mask[{i}]")
            axs[i][3].axis("off")
    else:
        kontexts = batch["kontext_images"].to(torch.float32)
        images   = batch["input_image"].to(torch.float32)
        masks    = batch["mask"].to(torch.float32)

        B = kontexts.shape[0]
        n_show = min(B, max_items)

        fig, axs = plt.subplots(n_show, 3, figsize=(figsize[0], figsize[1] * n_show / 3))
        if n_show == 1:
            axs = [axs]  # 保持二维结构

        for i in range(n_show):
            kontext = prepare_image(kontexts[i])
            image   = prepare_image(images[i])
            mask    = prepare_image(masks[i])
            axs[i][0].imshow(kontext)
            axs[i][0].set_title(f"kontext_images[{i}]")
            axs[i][0].axis("off")

            axs[i][1].imshow(image)
            axs[i][1].set_title(f"image[{i}]")
            axs[i][1].axis("off")

            axs[i][2].imshow(mask, cmap="gray")
            axs[i][2].set_title(f"mask[{i}]")
            axs[i][2].axis("off")

    plt.tight_layout()
    plt.savefig("batch.png")



class FluxImagePipeline(BasePipeline):

    def __init__(self, device="cuda", torch_dtype=torch.bfloat16):
        super().__init__(
            device=device, torch_dtype=torch_dtype,
            height_division_factor=16, width_division_factor=16,
        )
        self.scheduler = FlowMatchScheduler()
        self.prompter = FluxPrompter(tokenizer_1_path="./FLUX.1-Kontext-dev/tokenizer", tokenizer_2_path="./FLUX.1-Kontext-dev/tokenizer_2")
        self.text_encoder_1: SD3TextEncoder1 = None
        self.text_encoder_2: FluxTextEncoder2 = None
        self.dit: FluxDiT = None
        self.vae_decoder: FluxVAEDecoder = None
        self.vae_encoder: FluxVAEEncoder = None
        self.controlnet = None
        self.ipadapter_image_encoder = None
        self.qwenvl = None
        self.lora_patcher = None
        self.unit_runner = PipelineUnitRunner()
        self.in_iteration_models = ("dit","lora_patcher")
        self.units = [
            FluxImageUnit_ShapeChecker(),
            FluxImageUnit_NoiseInitializer(),
            FluxImageUnit_PromptEmbedder(),
            FluxImageUnit_InputImageEmbedder(),
            FluxImageUnit_ImageIDs(),
            FluxImageUnit_EmbeddedGuidanceEmbedder(),
            FluxImageUnit_Kontext(),
            FluxImageUnit_LoRAEncode(),
        ]
        self.model_fn = model_fn_flux_image
        self.loader = FluxLoRALoader(torch_dtype=self.torch_dtype, device=self.device)

    
    def training_loss(self, **inputs):
        # timestep_id = torch.randint(0, self.scheduler.num_train_timesteps, (1,))
        # timestep = self.scheduler.timesteps[timestep_id].to(dtype=self.torch_dtype, device=self.device)
        # timestep already include in inputs dict
        
        ##############################
        ##      flow_loss           ##
        ##############################
        # Forward diffusion (add noise)
        # Original Flow matching loss, latents z^y_t, input_latents z^x, noise z^y_0
        # currently z^y_t = (1-sigma) z^x + sigma * z_^y_0
        # However, it's not suitable for a deterministic task, like depth estimation or normal estimation
        # Now we want the latents to be independent of the timestep t, i.e. z^y_t = z^y_0
        if inputs.get("deterministic_flow", False):
            # inputs["latents"] = inputs["noise"] # current latents is just noise, no need to set again
            inputs["timestep"] = torch.full_like(inputs["timestep"], self.scheduler.timesteps[0]) # fixed timestep
            per_sample_weights = torch.ones_like(inputs["timestep"])
        else:
            inputs["latents"] = self.scheduler.add_noise(inputs["input_latents"], inputs["noise"], inputs["timestep"])
            per_sample_weights = self.scheduler.training_weight(inputs["timestep"])
            
        # if inputs.get("extra_loss", None) is None:
        training_target = self.scheduler.training_target(inputs["input_latents"], inputs["noise"], inputs["timestep"])

        # Predict target (vector field / noise difference)
        noise_pred = self.model_fn(**inputs)

        # Base (flow) loss (per-sample weighting)
        per_pixel_loss = torch.nn.functional.mse_loss(noise_pred.float(), training_target.float(), reduction="none")
        
        if inputs.get("mask", None) is not None:
            mask = repeat(inputs["mask"], 'b h w -> b c h w', c=1)
            per_pixel_mask_multiplier = torch.nn.functional.interpolate(mask.float(), size=inputs["latents"].shape[2:], mode="nearest").bool()
            per_pixel_loss = per_pixel_loss * per_pixel_mask_multiplier
            del mask
        per_pixel_loss = per_pixel_loss.mean(dim=[1,2,3])
        flow_loss = (per_pixel_loss * per_sample_weights).mean()
        # else:
        if inputs.get("extra_loss", None) is not None:
            # noise_pred = self.model_fn(**inputs)
            sigmas = self.scheduler.getsigmas(1000-inputs["timestep"],device=self.device,ndim=4).detach()
            pred = inputs["latents"].detach() + (sigmas - 1) * noise_pred
            pred = pred.to(dtype=self.torch_dtype)
            decoded = self.vae_decoder(pred, device=self.device)
            if inputs.get("extra_loss", None) == "cycle_consistency_normal_estimation":
                cycle_consistency_loss = get_cycle_consistency_normal_loss(decoded,inputs["input_image"], inputs.get("mask", None),per_sample_weights=per_sample_weights)
            elif inputs.get("extra_loss", None) == "cycle_consistency_depth_estimation":
                cycle_consistency_loss = get_cycle_consistency_depth_loss(decoded,inputs["input_image"], inputs.get("mask", None),depth_normalization=inputs.get("depth_normalization","log"))
            elif inputs.get("extra_loss",None) == "cycle_consistency_matting_estimation":
                cycle_consistency_loss = get_cycle_consistency_matting_loss(decoded,inputs["input_image"], inputs.get("trimap", None))
            # return flow_loss, cycle_consistency_loss
            return cycle_consistency_loss.to(torch.bfloat16)
        return flow_loss

    def _enable_vram_management_with_default_config(self, model, vram_limit):
        if model is not None:
            dtype = next(iter(model.parameters())).dtype
            enable_vram_management(
                model,
                module_map = {
                    torch.nn.Linear: AutoWrappedLinear,
                    torch.nn.Embedding: AutoWrappedModule,
                    torch.nn.LayerNorm: AutoWrappedModule,
                    torch.nn.Conv2d: AutoWrappedModule,
                    torch.nn.GroupNorm: AutoWrappedModule,
                    RMSNorm: AutoWrappedModule,
                    # LoRALayerBlock: AutoWrappedModule,
                },
                module_config = dict(
                    offload_dtype=dtype,
                    offload_device="cpu",
                    onload_dtype=dtype,
                    onload_device="cpu",
                    computation_dtype=self.torch_dtype,
                    computation_device=self.device,
                ),
                vram_limit=vram_limit,
            )
            
            
    def enable_lora_magic(self):
        if self.dit is not None:
            if not (hasattr(self.dit, "vram_management_enabled") and self.dit.vram_management_enabled):
                dtype = next(iter(self.dit.parameters())).dtype
                enable_vram_management(
                    self.dit,
                    module_map = {
                        torch.nn.Linear: AutoWrappedLinear,
                    },
                    module_config = dict(
                        offload_dtype=dtype,
                        offload_device=self.device,
                        onload_dtype=dtype,
                        onload_device=self.device,
                        computation_dtype=self.torch_dtype,
                        computation_device=self.device,
                    ),
                    vram_limit=None,
                )
        if self.lora_patcher is not None:
            for name, module in self.dit.named_modules():
                if isinstance(module, AutoWrappedLinear):
                    merger_name = name.replace(".", "___")
                    if merger_name in self.lora_patcher.model_dict:
                        module.lora_merger = self.lora_patcher.model_dict[merger_name]

        
    def load_lora(
        self,
        module: torch.nn.Module,
        lora_config: Union[ModelConfig, str] = None,
        alpha=1,
        hotload=False,
        state_dict=None,
    ):
        if state_dict is None:
            if isinstance(lora_config, str):
                lora = load_state_dict(lora_config, torch_dtype=self.torch_dtype, device=self.device)
            else:
                lora_config.download_if_necessary()
                lora = load_state_dict(lora_config.path, torch_dtype=self.torch_dtype, device=self.device)
        else:
            lora = state_dict
        lora = self.loader.convert_state_dict(lora)
        if hotload:
            for name, module in module.named_modules():
                if isinstance(module, AutoWrappedLinear):
                    lora_a_name = f'{name}.lora_A.default.weight'
                    lora_b_name = f'{name}.lora_B.default.weight'
                    if lora_a_name in lora and lora_b_name in lora:
                        module.lora_A_weights.append(lora[lora_a_name] * alpha)
                        module.lora_B_weights.append(lora[lora_b_name])
        else:
            self.loader.load(module, lora, alpha=alpha)


    def load_loras(
        self,
        module: torch.nn.Module,
        lora_configs: list[Union[ModelConfig, str]],
        alpha=1,
        hotload=False,
        extra_fused_lora=False,
    ):
        for lora_config in lora_configs:
            self.load_lora(module, lora_config, hotload=hotload, alpha=alpha)
        if extra_fused_lora:
            lora_fuser = FluxLoRAFuser(device="cuda", torch_dtype=torch.bfloat16)
            fused_lora = lora_fuser(lora_configs)
            self.load_lora(module, state_dict=fused_lora, hotload=hotload, alpha=alpha)

    
    def clear_lora(self):
        for name, module in self.named_modules():
            if isinstance(module, AutoWrappedLinear): 
                if hasattr(module, "lora_A_weights"):
                    module.lora_A_weights.clear()
                if hasattr(module, "lora_B_weights"):
                    module.lora_B_weights.clear()
    
    
    def enable_vram_management(self, num_persistent_param_in_dit=None, vram_limit=None, vram_buffer=0.5):
        self.vram_management_enabled = True
        if num_persistent_param_in_dit is not None:
            vram_limit = None
        else:
            if vram_limit is None:
                vram_limit = self.get_vram()
            vram_limit = vram_limit - vram_buffer

        # Default config
        # default_vram_management_models = ["text_encoder_1", "vae_decoder", "vae_encoder", "controlnet", "image_proj_model", "ipadapter", "lora_patcher", "value_controller", "step1x_connector", "lora_encoder"]
        default_vram_management_models = ["text_encoder_1", "vae_decoder", "vae_encoder"]
        for model_name in default_vram_management_models:
            self._enable_vram_management_with_default_config(getattr(self, model_name), vram_limit)

        # Special config
        if self.text_encoder_2 is not None:
            from transformers.models.t5.modeling_t5 import T5LayerNorm, T5DenseActDense, T5DenseGatedActDense
            dtype = next(iter(self.text_encoder_2.parameters())).dtype
            enable_vram_management(
                self.text_encoder_2,
                module_map = {
                    torch.nn.Linear: AutoWrappedLinear,
                    torch.nn.Embedding: AutoWrappedModule,
                    T5LayerNorm: AutoWrappedModule,
                    T5DenseActDense: AutoWrappedModule,
                    T5DenseGatedActDense: AutoWrappedModule,
                },
                module_config = dict(
                    offload_dtype=dtype,
                    offload_device="cpu",
                    onload_dtype=dtype,
                    onload_device="cpu",
                    computation_dtype=self.torch_dtype,
                    computation_device=self.device,
                ),
                vram_limit=vram_limit,
            )
        if self.dit is not None:
            dtype = next(iter(self.dit.parameters())).dtype
            device = "cpu" if vram_limit is not None else self.device
            enable_vram_management(
                self.dit,
                module_map = {
                    RMSNorm: AutoWrappedModule,
                    torch.nn.Linear: AutoWrappedLinear,
                },
                module_config = dict(
                    offload_dtype=dtype,
                    offload_device="cpu",
                    onload_dtype=dtype,
                    onload_device=device,
                    computation_dtype=self.torch_dtype,
                    computation_device=self.device,
                ),
                max_num_param=num_persistent_param_in_dit,
                overflow_module_config = dict(
                    offload_dtype=dtype,
                    offload_device="cpu",
                    onload_dtype=dtype,
                    onload_device="cpu",
                    computation_dtype=self.torch_dtype,
                    computation_device=self.device,
                ),
                vram_limit=vram_limit,
            )
        if self.ipadapter_image_encoder is not None:
            from transformers.models.siglip.modeling_siglip import SiglipVisionEmbeddings, SiglipEncoder, SiglipMultiheadAttentionPoolingHead
            dtype = next(iter(self.ipadapter_image_encoder.parameters())).dtype
            enable_vram_management(
                self.ipadapter_image_encoder,
                module_map = {
                    SiglipVisionEmbeddings: AutoWrappedModule,
                    SiglipEncoder: AutoWrappedModule,
                    SiglipMultiheadAttentionPoolingHead: AutoWrappedModule,
                    torch.nn.MultiheadAttention: AutoWrappedModule,
                    torch.nn.Linear: AutoWrappedLinear,
                    torch.nn.LayerNorm: AutoWrappedModule,
                },
                module_config = dict(
                    offload_dtype=dtype,
                    offload_device="cpu",
                    onload_dtype=dtype,
                    onload_device="cpu",
                    computation_dtype=self.torch_dtype,
                    computation_device=self.device,
                ),
                vram_limit=vram_limit,
            )
        if self.qwenvl is not None:
            from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
                Qwen2_5_VisionPatchEmbed, Qwen2_5_VLVisionBlock, Qwen2_5_VLPatchMerger,
                Qwen2_5_VLDecoderLayer, Qwen2_5_VisionRotaryEmbedding, Qwen2_5_VLRotaryEmbedding, Qwen2RMSNorm
            )
            dtype = next(iter(self.qwenvl.parameters())).dtype
            enable_vram_management(
                self.qwenvl,
                module_map = {
                    Qwen2_5_VisionPatchEmbed: AutoWrappedModule,
                    Qwen2_5_VLVisionBlock: AutoWrappedModule,
                    Qwen2_5_VLPatchMerger: AutoWrappedModule,
                    Qwen2_5_VLDecoderLayer: AutoWrappedModule,
                    Qwen2_5_VisionRotaryEmbedding: AutoWrappedModule,
                    Qwen2_5_VLRotaryEmbedding: AutoWrappedModule,
                    Qwen2RMSNorm: AutoWrappedModule,
                    torch.nn.Embedding: AutoWrappedModule,
                    torch.nn.Linear: AutoWrappedLinear,
                    torch.nn.LayerNorm: AutoWrappedModule,
                },
                module_config = dict(
                    offload_dtype=dtype,
                    offload_device="cpu",
                    onload_dtype=dtype,
                    onload_device="cpu",
                    computation_dtype=self.torch_dtype,
                    computation_device=self.device,
                ),
                vram_limit=vram_limit,
            )
    
    
    @staticmethod
    def from_pretrained(
        torch_dtype: torch.dtype = torch.bfloat16,
        device: Union[str, torch.device] = "cuda",
        model_configs: list[ModelConfig] = [],
        finetuned_dit_path: Optional[str] = None,
    ):
        # Download and load models
        model_manager = ModelManager()
        for model_config in model_configs:
            model_config.download_if_necessary()
            model_manager.load_model(
                model_config.path,
                device=model_config.offload_device or device,
                torch_dtype=model_config.offload_dtype or torch_dtype
            )
        
        # Load finetuned DiT if provided (merged weights)
        if finetuned_dit_path is not None:
            print(f"Loading finetuned DiT from: {finetuned_dit_path}")
            model_manager.load_model(
                finetuned_dit_path,
                device=device,
                torch_dtype=torch_dtype
            )
        
        # Initialize pipeline
        pipe = FluxImagePipeline(device=device, torch_dtype=torch_dtype)
        pipe.text_encoder_1 = model_manager.fetch_model("sd3_text_encoder_1")
        # pipe.text_encoder_1 = model_manager.fetch_model("clip_text_encoder")
        pipe.text_encoder_2 = model_manager.fetch_model("flux_text_encoder_2")
        # Safety: ensure text encoders are on the main pipeline device (some loader paths may leave them on cpu)
        if pipe.text_encoder_1 is not None:
            try:
                pipe.text_encoder_1.to(device=device)
            except Exception as _e:
                pass
        if pipe.text_encoder_2 is not None:
            try:
                pipe.text_encoder_2.to(device=device)
            except Exception as _e:
                pass
        
        pipe.dit =  model_manager.fetch_model("flux_dit")
        # pipe.dit = model_manager.fetch_model("flux_transformer2dmodel")
        pipe.vae_decoder = model_manager.fetch_model("flux_vae_decoder")
        pipe.vae_encoder = model_manager.fetch_model("flux_vae_encoder")
        # Ensure core diffusion components are on target device (avoid CPU weights during first forward)
        for _m in [pipe.dit, pipe.vae_decoder, pipe.vae_encoder]:
            if _m is not None:
                try:
                    _m.to(device=device)
                except Exception:
                    pass
        pipe.prompter.fetch_models(pipe.text_encoder_1, pipe.text_encoder_2)

        pipe.lora_patcher = model_manager.fetch_model("flux_lora_patcher")
        pipe.lora_encoder = model_manager.fetch_model("flux_lora_encoder")
        # print which ones are Loaded (not None)
        _models = {"text_encoder_1","text_encoder_2","dit","vae_decoder","vae_encoder"}
        for _model in _models:
            if getattr(pipe, _model, None) is not None:
                print(f"Loaded {_model}")
        
        pipe._device_normalized = True
        return pipe
    
    
    @torch.no_grad()
    def __call__(
        self,
        # Prompt
        prompt: str,
        negative_prompt: str = "",
        cfg_scale: float = 1.0,
        embedded_guidance: float = 3.5,
        t5_sequence_length: int = 512,
        # Image: PIL Image or Tensor
        input_image= None,
        denoising_strength: float = 1.0,
        # Shape
        height: int = 1024,
        width: int = 1024,
        # Randomness
        seed: int = 42,
        rand_device: str = "cpu",
        # Scheduler
        sigma_shift: float = 0.5,
        dynamic_shift_len: int = None,
        # Steps
        num_inference_steps: int = 30,
        # local prompts
        multidiffusion_prompts=(),
        multidiffusion_masks=(),
        multidiffusion_scales=(),
        # Kontext
        kontext_images: Union[list[Image.Image], Image.Image, torch.Tensor] = None,
        # LoRA Encoder
        lora_encoder_inputs: Union[list[ModelConfig], ModelConfig, str] = None,
        lora_encoder_scale: float = 1.0,
        # TeaCache
        tea_cache_l1_thresh: float = None,
        # Tile
        tiled: bool = False,
        tile_size: int = 128,
        tile_stride: int = 64,
        # Progress bar
        progress_bar_cmd = tqdm,
        output_type: Literal["pil", "np"] = "pil",
        task: Literal["depth", "normal"] = "depth",
        multi_res_noise: bool = False,
        deterministic_flow: bool = False,
        kontext_masks = None,
    ):
        # Scheduler
        
        inputs_posi = {
            "prompt": prompt,
        }
        inputs_nega = {
            "negative_prompt": negative_prompt,
        }
        inputs_shared = {
            "cfg_scale": cfg_scale, "embedded_guidance": embedded_guidance, "t5_sequence_length": t5_sequence_length,
            "input_image": input_image, "denoising_strength": denoising_strength,
            "height": height, "width": width,
            "seed": seed, "rand_device": rand_device,
            "sigma_shift": sigma_shift, "num_inference_steps": num_inference_steps,
            "multidiffusion_prompts": multidiffusion_prompts, "multidiffusion_masks": multidiffusion_masks, "multidiffusion_scales": multidiffusion_scales,
            "kontext_images": kontext_images,
            "lora_encoder_inputs": lora_encoder_inputs, "lora_encoder_scale": lora_encoder_scale,
            "tea_cache_l1_thresh": tea_cache_l1_thresh,
            "tiled": tiled, "tile_size": tile_size, "tile_stride": tile_stride,
            "progress_bar_cmd": progress_bar_cmd,
            "task": task,
            "multi_res_noise": multi_res_noise, 
            "kontext_masks": kontext_masks,
            "deterministic_flow": deterministic_flow,
        }

        # Denoise
        self.load_models_to_device(self.in_iteration_models)
        models = {name: getattr(self, name) for name in self.in_iteration_models}
        if deterministic_flow or num_inference_steps == 1:
        # if deterministic_flow:
            self.scheduler.set_timesteps(1000, training=True)
            timestep = self.scheduler.timesteps[0].unsqueeze(0).to(dtype=self.torch_dtype, device=self.device)
            inputs_shared["timestep"] = timestep
            # inputs_shared["multi_res_noise"] = True
            for unit in self.units:
                inputs_shared, inputs_posi, inputs_nega = self.unit_runner(unit, self, inputs_shared, inputs_posi, inputs_nega)
            # Inference
            noise_pred = self.model_fn(**models, **inputs_shared, **inputs_posi)
            sigmas = self.scheduler.getsigmas(1000-timestep.to(torch.float32), device=self.device, ndim=4)
            inputs_shared["latents"] = inputs_shared["latents"] + (sigmas - 1) * noise_pred
        else:
            self.scheduler.set_timesteps(num_inference_steps, training=True)

            for unit in self.units:
                inputs_shared, inputs_posi, inputs_nega = self.unit_runner(unit, self, inputs_shared, inputs_posi, inputs_nega)

            for progress_id, timestep in enumerate(progress_bar_cmd(self.scheduler.timesteps,disable=True)):
                timestep = timestep.unsqueeze(0).to(dtype=self.torch_dtype, device=self.device)

                # Inference
                noise_pred_posi = self.model_fn(**models, **inputs_shared, **inputs_posi, timestep=timestep, progress_id=progress_id)
                if cfg_scale != 1.0:
                    noise_pred_nega = self.model_fn(**models, **inputs_shared, **inputs_nega, timestep=timestep, progress_id=progress_id)
                    noise_pred = noise_pred_nega + cfg_scale * (noise_pred_posi - noise_pred_nega)
                else:
                    noise_pred = noise_pred_posi
                # Scheduler
                inputs_shared["latents"] = self.scheduler.step(noise_pred, self.scheduler.timesteps[progress_id].unsqueeze(0), inputs_shared["latents"],to_final=(progress_id == len(self.scheduler.timesteps) - 1))
        
        # Decode
        self.load_models_to_device(['vae_decoder'])
        image = self.vae_decoder(inputs_shared["latents"].to(self.torch_dtype), device=self.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        if output_type == "np":
            image = self.vae_out_to_np(image,task=task)
        else:
            image = self.vae_output_to_image(image)
        self.load_models_to_device([])

        return image

        

class FluxImageUnit_ShapeChecker(PipelineUnit):
    def __init__(self):
        super().__init__(input_params=("height", "width","kontext_images"))

    def process(self, pipe: FluxImagePipeline, height, width, kontext_images):
        height, width = pipe.check_resize_height_width(height, width)
        if isinstance(kontext_images, torch.Tensor) and kontext_images.ndim == 3:
            kontext_images = kontext_images.unsqueeze(0)
        elif isinstance(kontext_images, list) and isinstance(kontext_images[0], torch.Tensor) and kontext_images[0].ndim == 3:
            kontext_images = [img.unsqueeze(0) for img in kontext_images]
        return {"height": height, "width": width, "kontext_images": kontext_images}



class FluxImageUnit_NoiseInitializer(PipelineUnit):
    def __init__(self):
        super().__init__(input_params=("kontext_images", "seed", "rand_device", "multi_res_noise","timestep","deterministic_flow"))

    def process(self, pipe: FluxImagePipeline, kontext_images, seed, rand_device, multi_res_noise=False, timestep=None,deterministic_flow=False):
        if timestep is None or multi_res_noise==False:
            multi_res_noise = False
            multi_res_noise_strength = None
        elif multi_res_noise:
            multi_res_noise_strength =(0.9*timestep/1000).to(rand_device)
        if isinstance(kontext_images, torch.Tensor) and kontext_images.ndim == 4:
            B, _, height, width = kontext_images.shape
        elif isinstance(kontext_images, Image.Image):
            B = 1
            height, width = kontext_images.height, kontext_images.width
        elif isinstance(kontext_images, list) and isinstance(kontext_images[0], torch.Tensor):
            B, _, height, width = kontext_images[0].shape
        if deterministic_flow:
            return {"noise": torch.zeros((B, 16, height//8, width//8), device=rand_device, dtype=pipe.torch_dtype)}
        # print(f"Generating noise of shape ({B}, 16, {height//8}, {width//8})")
        noise = pipe.generate_noise(
            (B, 16, height//8, width//8), 
            seed=seed, 
            rand_device=rand_device, 
            multi_res_noise=multi_res_noise,
            multi_res_noise_strength=multi_res_noise_strength
        )
        return {"noise": noise}

class FluxImageUnit_InputImageEmbedder(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("input_image", "noise", "tiled", "tile_size", "tile_stride"),
            onload_model_names=("vae_encoder",)
        )

    def process(self, pipe: FluxImagePipeline, input_image, noise, tiled, tile_size, tile_stride):
        if input_image is None:
            return {"latents": noise, "input_latents": None}
        pipe.load_models_to_device(['vae_encoder'])
        image = pipe.preprocess_image(input_image).to(device=pipe.device, dtype=pipe.torch_dtype)
        input_latents = pipe.vae_encoder(image, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        if pipe.scheduler.training:
            return {"latents": noise, "input_latents": input_latents}
        else:
            latents = pipe.scheduler.add_noise(input_latents, noise, timestep=pipe.scheduler.timesteps[0])
            return {"latents": latents, "input_latents": None}



class FluxImageUnit_PromptEmbedder(PipelineUnit):
    def __init__(self):
        super().__init__(
            seperate_cfg=True,
            input_params_posi={"prompt": "prompt", "positive": "positive"},
            input_params_nega={"prompt": "negative_prompt", "positive": "positive"},
            input_params=("t5_sequence_length",),
            onload_model_names=("text_encoder_1", "text_encoder_2")
        )

    def process(self, pipe: FluxImagePipeline, prompt, t5_sequence_length, positive) -> dict:
        if pipe.text_encoder_1 is not None and pipe.text_encoder_2 is not None:
            prompt_emb, pooled_prompt_emb, text_ids = pipe.prompter.encode_prompt(
                prompt, device=pipe.device, positive=positive, t5_sequence_length=t5_sequence_length
            )
            return {"prompt_emb": prompt_emb, "pooled_prompt_emb": pooled_prompt_emb, "text_ids": text_ids}
        else:
            return {}


class FluxImageUnit_ImageIDs(PipelineUnit):
    def __init__(self):
        super().__init__(input_params=("latents",))

    def process(self, pipe: FluxImagePipeline, latents):
        latent_image_ids = pipe.dit.prepare_image_ids(latents)
        return {"image_ids": latent_image_ids}



class FluxImageUnit_EmbeddedGuidanceEmbedder(PipelineUnit):
    def __init__(self):
        super().__init__(input_params=("embedded_guidance", "latents"))

    def process(self, pipe: FluxImagePipeline, embedded_guidance, latents):
        guidance = torch.Tensor([embedded_guidance] * latents.shape[0]).to(device=latents.device, dtype=latents.dtype)
        return {"guidance": guidance}



class FluxImageUnit_Kontext(PipelineUnit):
    def __init__(self):
        super().__init__(input_params=("kontext_images", "tiled", "tile_size", "tile_stride"))

    def process(self, pipe: FluxImagePipeline, kontext_images, tiled, tile_size, tile_stride):
        if kontext_images is None:
            return {}
        if isinstance(kontext_images, torch.Tensor):
            kontext_images = pipe.preprocess_image(kontext_images)
            kontext_latents = pipe.vae_encoder(kontext_images, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
            kontext_image_ids = pipe.dit.prepare_image_ids(kontext_latents)
            kontext_image_ids[..., 0] = 1
            kontext_latents = pipe.dit.patchify(kontext_latents)
        elif not isinstance(kontext_images, list) and isinstance(kontext_images, Image.Image):
            kontext_images = [kontext_images]
            kontext_latents = []
            kontext_image_ids = []
            for kontext_image in kontext_images:
                kontext_image = pipe.preprocess_image(kontext_image)
                kontext_latent = pipe.vae_encoder(kontext_image, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
                image_ids = pipe.dit.prepare_image_ids(kontext_latent)
                image_ids[..., 0] = 1
                kontext_image_ids.append(image_ids)
                kontext_latent = pipe.dit.patchify(kontext_latent)
                kontext_latents.append(kontext_latent)
            kontext_latents = torch.concat(kontext_latents, dim=1)
            kontext_image_ids = torch.concat(kontext_image_ids, dim=-2)
        elif isinstance(kontext_images, list) and isinstance(kontext_images[0], torch.Tensor):
            kontext_latents = []
            kontext_image_ids = []
            for kontext_image in kontext_images:
                kontext_image = pipe.preprocess_image(kontext_image)
                kontext_latent = pipe.vae_encoder(kontext_image, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
                image_ids = pipe.dit.prepare_image_ids(kontext_latent)
                image_ids[..., 0] = 1
                kontext_image_ids.append(image_ids)
                kontext_latent = pipe.dit.patchify(kontext_latent)
                kontext_latents.append(kontext_latent)
            kontext_latents = torch.concat(kontext_latents, dim=1)
            kontext_image_ids = torch.concat(kontext_image_ids, dim=-2)
        return {"kontext_latents": kontext_latents, "kontext_image_ids": kontext_image_ids}



class FluxImageUnit_LoRAEncode(PipelineUnit):
    def __init__(self):
        super().__init__(
            take_over=True,
            onload_model_names=("lora_encoder",)
        )
        
    def parse_lora_encoder_inputs(self, lora_encoder_inputs):
        if not isinstance(lora_encoder_inputs, list):
            lora_encoder_inputs = [lora_encoder_inputs]
        lora_configs = []
        for lora_encoder_input in lora_encoder_inputs:
            if isinstance(lora_encoder_input, str):
                lora_encoder_input = ModelConfig(path=lora_encoder_input)
            lora_encoder_input.download_if_necessary()
            lora_configs.append(lora_encoder_input)
        return lora_configs
        
    def load_lora(self, lora_config, dtype, device):
        loader = FluxLoRALoader(torch_dtype=dtype, device=device)
        lora = load_state_dict(lora_config.path, torch_dtype=dtype, device=device)
        lora = loader.convert_state_dict(lora)
        return lora
    
    def lora_embedding(self, pipe, lora_encoder_inputs):
        lora_emb = []
        for lora_config in self.parse_lora_encoder_inputs(lora_encoder_inputs):
            lora = self.load_lora(lora_config, pipe.torch_dtype, pipe.device)
            lora_emb.append(pipe.lora_encoder(lora))
        lora_emb = torch.concat(lora_emb, dim=1)
        return lora_emb
    
    def add_to_text_embedding(self, prompt_emb, text_ids, lora_emb):
        prompt_emb = torch.concat([prompt_emb, lora_emb], dim=1)
        extra_text_ids = torch.zeros((lora_emb.shape[0], lora_emb.shape[1], 3), device=lora_emb.device, dtype=lora_emb.dtype)
        text_ids = torch.concat([text_ids, extra_text_ids], dim=1)
        return prompt_emb, text_ids

    def process(self, pipe: FluxImagePipeline, inputs_shared, inputs_posi, inputs_nega):
        if inputs_shared.get("lora_encoder_inputs", None) is None:
            return inputs_shared, inputs_posi, inputs_nega
        
        # Encode
        pipe.load_models_to_device(["lora_encoder"])
        lora_encoder_inputs = inputs_shared["lora_encoder_inputs"]
        lora_emb = self.lora_embedding(pipe, lora_encoder_inputs)
        
        # Scale
        lora_encoder_scale = inputs_shared.get("lora_encoder_scale", None)
        if lora_encoder_scale is not None:
            lora_emb = lora_emb * lora_encoder_scale
        
        # Add to prompt embedding
        inputs_posi["prompt_emb"], inputs_posi["text_ids"] = self.add_to_text_embedding(
            inputs_posi["prompt_emb"], inputs_posi["text_ids"], lora_emb)
        return inputs_shared, inputs_posi, inputs_nega



class TeaCache:
    def __init__(self, num_inference_steps, rel_l1_thresh):
        self.num_inference_steps = num_inference_steps
        self.step = 0
        self.accumulated_rel_l1_distance = 0
        self.previous_modulated_input = None
        self.rel_l1_thresh = rel_l1_thresh
        self.previous_residual = None
        self.previous_hidden_states = None

    def check(self, dit: FluxDiT, hidden_states, conditioning):
        inp = hidden_states.clone()
        temb_ = conditioning.clone()
        modulated_inp, _, _, _, _ = dit.blocks[0].norm1_a(inp, emb=temb_)
        if self.step == 0 or self.step == self.num_inference_steps - 1:
            should_calc = True
            self.accumulated_rel_l1_distance = 0
        else: 
            coefficients = [4.98651651e+02, -2.83781631e+02,  5.58554382e+01, -3.82021401e+00, 2.64230861e-01]
            rescale_func = np.poly1d(coefficients)
            self.accumulated_rel_l1_distance += rescale_func(((modulated_inp-self.previous_modulated_input).abs().mean() / self.previous_modulated_input.abs().mean()).cpu().item())
            if self.accumulated_rel_l1_distance < self.rel_l1_thresh:
                should_calc = False
            else:
                should_calc = True
                self.accumulated_rel_l1_distance = 0
        self.previous_modulated_input = modulated_inp 
        self.step += 1
        if self.step == self.num_inference_steps:
            self.step = 0
        if should_calc:
            self.previous_hidden_states = hidden_states.clone()
        return not should_calc
    
    def store(self, hidden_states):
        self.previous_residual = hidden_states - self.previous_hidden_states
        self.previous_hidden_states = None

    def update(self, hidden_states):
        hidden_states = hidden_states + self.previous_residual
        return hidden_states
    
    
def model_fn_flux_image(
    dit: FluxDiT,
    controlnet=None,
    latents=None,
    timestep=None,
    prompt_emb=None,
    pooled_prompt_emb=None,
    guidance=None,
    text_ids=None,
    image_ids=None,
    kontext_latents=None,
    kontext_image_ids=None,
    controlnet_inputs=None,
    controlnet_conditionings=None,
    tiled=False,
    tile_size=128,
    tile_stride=64,
    entity_prompt_emb=None,
    entity_masks=None,
    ipadapter_kwargs_list={},
    progress_id=0,
    num_inference_steps=1,
    use_gradient_checkpointing=False,
    use_gradient_checkpointing_offload=False,
    visual_prompt_coords=None,
    kontext_masks=None,
    **kwargs
):
    if tiled:
        def flux_forward_fn(hl, hr, wl, wr):
            tiled_controlnet_conditionings = [f[:, :, hl: hr, wl: wr] for f in controlnet_conditionings] if controlnet_conditionings is not None else None
            return model_fn_flux_image(
                dit=dit,
                controlnet=controlnet,
                latents=latents[:, :, hl: hr, wl: wr],
                timestep=timestep,
                prompt_emb=prompt_emb,
                pooled_prompt_emb=pooled_prompt_emb,
                guidance=guidance,
                text_ids=text_ids,
                image_ids=None,
                controlnet_inputs=controlnet_inputs,
                controlnet_conditionings=tiled_controlnet_conditionings,
                tiled=False,
                visual_prompt_coords=visual_prompt_coords,
                **kwargs
            )
        return FastTileWorker().tiled_forward(
            flux_forward_fn,
            latents,
            tile_size=tile_size,
            tile_stride=tile_stride,
            tile_device=latents.device,
            tile_dtype=latents.dtype
        )

    hidden_states = latents



    if image_ids is None:
        image_ids = dit.prepare_image_ids(hidden_states)
    
    conditioning = dit.time_embedder(timestep, hidden_states.dtype) + dit.pooled_text_embedder(pooled_prompt_emb)
    if dit.guidance_embedder is not None:
        guidance = guidance * 1000
        conditioning = conditioning + dit.guidance_embedder(guidance, hidden_states.dtype)
    
    # Compute coordinate embeddings if enabled
    if dit.use_coor_input and visual_prompt_coords is not None:
        # Call coord_encoder to get embeddings (1280-dim)
        coord_emb = dit.coord_encoder(visual_prompt_coords.to(torch.bfloat16))
        # Project to conditioning dimension (3072-dim) and add to conditioning
        coord_emb = dit.coord_proj(coord_emb)
        coord_emb = coord_emb.to(conditioning.dtype)
        conditioning = conditioning + coord_emb

    height, width = hidden_states.shape[-2:]
    hidden_states = dit.patchify(hidden_states)
    
    # Kontext
    if kontext_latents is not None:
        image_ids = torch.concat([image_ids, kontext_image_ids], dim=-2)
        hidden_states = torch.concat([hidden_states, kontext_latents], dim=1)
    
    hidden_states = hidden_states.to(timestep.dtype)
    hidden_states = dit.x_embedder(hidden_states)


    prompt_emb = dit.context_embedder(prompt_emb)
    image_rotary_emb = dit.pos_embedder(torch.cat((text_ids, image_ids), dim=1))
    # Joint Blocks
    if kontext_masks is not None:
        attention_mask = dit.process_kontext_attention_mask(kontext_masks,kontext_image_ids,prompt_seq_len=512).unsqueeze(1)
    else:
        attention_mask = None
    for block_id, block in enumerate(dit.blocks):
        hidden_states, prompt_emb = gradient_checkpoint_forward(
            block,
            use_gradient_checkpointing,
            use_gradient_checkpointing_offload,
            hidden_states,
            prompt_emb,
            conditioning,
            image_rotary_emb,
            attention_mask,
            ipadapter_kwargs_list=ipadapter_kwargs_list.get(block_id, None),
        )
    # Single Blocks
    hidden_states = torch.cat([prompt_emb, hidden_states], dim=1)
    num_joint_blocks = len(dit.blocks)
    if kontext_masks is not None:
        # attention_mask = dit.process_kontext_attention_mask(kontext_masks,kontext_image_ids,base_image_seq_len=latents.shape[1],prompt_seq_len=prompt_emb.shape[1])
        attention_mask = dit.process_kontext_attention_mask(kontext_masks,kontext_image_ids,prompt_seq_len=512).unsqueeze(1)
    else:
        attention_mask = None
    for block_id, block in enumerate(dit.single_blocks):
        hidden_states, prompt_emb = gradient_checkpoint_forward(
            block,
            use_gradient_checkpointing,
            use_gradient_checkpointing_offload,
            hidden_states,
            prompt_emb,
            conditioning,
            image_rotary_emb,
            attention_mask,
            ipadapter_kwargs_list=ipadapter_kwargs_list.get(block_id + num_joint_blocks, None),
        )
    hidden_states = hidden_states[:, prompt_emb.shape[1]:]

    hidden_states = dit.final_norm_out(hidden_states, conditioning)
    hidden_states = dit.final_proj_out(hidden_states)
    
    
    # Kontext
    if kontext_latents is not None:
        hidden_states = hidden_states[:, :-kontext_latents.shape[1]]

    hidden_states = dit.unpatchify(hidden_states, height, width)

    return hidden_states