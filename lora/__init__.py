import torch
from typing import Dict, Optional, Tuple


class GeneralLoRALoader:
    """
    支持动态加载和卸载 LoRA 的加载器（显存优化版）。
    
    核心机制：
    1. 只保存 LoRA 的原始 A/B 矩阵（存储在 CPU 上）
    2. 加载/卸载时临时计算增量，用完即释放
    3. 切换 LoRA 时，先卸载旧的，再加载新的
    
    用法:
        loader = GeneralLoRALoader(device="cuda", torch_dtype=torch.bfloat16)
        
        # 加载 LoRA
        loader.load(model, lora_state_dict, alpha=1.0)
        
        # 卸载当前 LoRA (恢复基础模型)
        loader.unload(model)
        
        # 切换到另一个 LoRA
        loader.switch(model, new_lora_state_dict, alpha=1.0)
    """
    
    def __init__(self, device="cpu", torch_dtype=torch.float32):
        self.device = device
        self.torch_dtype = torch_dtype
        # 存储当前加载的 LoRA 原始矩阵 (存在 CPU 上节省显存): 
        # {module_name: (weight_up, weight_down)}
        self._current_lora_weights: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
        # 当前加载的 LoRA 的 alpha 值
        self._current_alpha: float = 0.0
        # 标记是否有 LoRA 被加载
        self._lora_loaded: bool = False
    
    
    def get_name_dict(self, lora_state_dict):
        lora_name_dict = {}
        for key in lora_state_dict:
            if ".lora_B." not in key:
                continue
            keys = key.split(".")
            if len(keys) > keys.index("lora_B") + 2:
                keys.pop(keys.index("lora_B") + 1)
            keys.pop(keys.index("lora_B"))
            if keys[0] == "diffusion_model":
                keys.pop(0)
            keys.pop(-1)
            target_name = ".".join(keys)
            lora_name_dict[target_name] = (key, key.replace(".lora_B.", ".lora_A."))
        return lora_name_dict

    def _compute_lora_delta(self, weight_up: torch.Tensor, weight_down: torch.Tensor, alpha: float) -> torch.Tensor:
        """计算 LoRA 权重增量"""
        if len(weight_up.shape) == 4:
            weight_up = weight_up.squeeze(3).squeeze(2)
            weight_down = weight_down.squeeze(3).squeeze(2)
            weight_lora = alpha * torch.mm(weight_up, weight_down).unsqueeze(2).unsqueeze(3)
        else:
            weight_lora = alpha * torch.mm(weight_up, weight_down)
        return weight_lora

    def load(self, model: torch.nn.Module, state_dict_lora, alpha=1.0):
        """
        加载 LoRA 权重到模型。
        如果已有 LoRA 被加载，会先自动卸载。
        """
        # 如果已经有 LoRA 加载，先卸载
        if self._lora_loaded:
            print("Detected existing LoRA, unloading first...")
            self.unload(model)
        
        updated_num = 0
        lora_name_dict = self.get_name_dict(state_dict_lora)
        self._current_lora_weights.clear()
        
        for name, module in model.named_modules():
            if name in lora_name_dict:
                weight_up = state_dict_lora[lora_name_dict[name][0]].to(dtype=self.torch_dtype)
                weight_down = state_dict_lora[lora_name_dict[name][1]].to(dtype=self.torch_dtype)
                
                # 保存原始 A/B 矩阵到 CPU（节省显存）
                self._current_lora_weights[name] = (
                    weight_up.cpu().clone(),
                    weight_down.cpu().clone()
                )
                
                # 临时移到 GPU 计算增量
                weight_up_gpu = weight_up.to(device=self.device)
                weight_down_gpu = weight_down.to(device=self.device)
                weight_lora = self._compute_lora_delta(weight_up_gpu, weight_down_gpu, alpha)
                
                # 应用到模型
                state_dict = module.state_dict()
                state_dict["weight"] = state_dict["weight"].to(device=self.device, dtype=self.torch_dtype) + weight_lora
                module.load_state_dict(state_dict)
                
                # 立即释放临时 GPU 张量
                del weight_up_gpu, weight_down_gpu, weight_lora
                updated_num += 1
        
        self._current_alpha = alpha
        self._lora_loaded = True
        
        # 清理 GPU 缓存
        if self.device != "cpu":
            torch.cuda.empty_cache()
        
        print(f"{updated_num} tensors are updated by LoRA.")
    
    def unload(self, model: torch.nn.Module):
        """
        卸载当前 LoRA，恢复基础模型权重。
        """
        if not self._lora_loaded:
            print("No LoRA is currently loaded.")
            return
        
        unloaded_num = 0
        for name, module in model.named_modules():
            if name in self._current_lora_weights:
                weight_up, weight_down = self._current_lora_weights[name]
                
                # 临时移到 GPU 计算增量
                weight_up_gpu = weight_up.to(device=self.device, dtype=self.torch_dtype)
                weight_down_gpu = weight_down.to(device=self.device, dtype=self.torch_dtype)
                weight_delta = self._compute_lora_delta(weight_up_gpu, weight_down_gpu, self._current_alpha)
                
                # 减去增量恢复原始权重
                state_dict = module.state_dict()
                state_dict["weight"] = state_dict["weight"].to(device=self.device, dtype=self.torch_dtype) - weight_delta
                module.load_state_dict(state_dict)
                
                # 立即释放临时 GPU 张量
                del weight_up_gpu, weight_down_gpu, weight_delta
                unloaded_num += 1
        
        self._current_lora_weights.clear()
        self._current_alpha = 0.0
        self._lora_loaded = False
        
        # 清理 GPU 缓存
        if self.device != "cpu":
            torch.cuda.empty_cache()
        
        print(f"{unloaded_num} tensors restored to base model.")
    
    def switch(self, model: torch.nn.Module, new_state_dict_lora, alpha=1.0):
        """
        快速切换到另一个 LoRA。
        等价于 unload() + load()，但语义更清晰。
        """
        if self._lora_loaded:
            self.unload(model)
        self.load(model, new_state_dict_lora, alpha)
    
    def is_loaded(self) -> bool:
        """检查是否有 LoRA 被加载"""
        return self._lora_loaded
    
    def get_loaded_modules(self) -> list:
        """获取当前加载了 LoRA 的模块名称列表"""
        return list(self._current_lora_weights.keys())
