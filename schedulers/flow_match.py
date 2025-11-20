import math
import torch
import numpy as np
from typing import Union, Optional
from diffusers import FlowMatchEulerDiscreteScheduler


def calculate_shift(
    image_seq_len,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.16,
):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu


class FlowMatchScheduler(FlowMatchEulerDiscreteScheduler):
    def __init__(
        self,
        num_inference_steps=100,
        num_train_timesteps=1000,
        shift=3.0,
        sigma_max=1.0,
        sigma_min=0.003/1.002,
        inverse_timesteps=False,
        extra_one_step=False,
        reverse_sigmas=False,
        exponential_shift=False,
        exponential_shift_mu=None,
        shift_terminal=None,
        # 新增的配置参数（与scheduler_config对应）
        base_image_seq_len=256,
        base_shift=0.5,
        max_image_seq_len=4096,
        max_shift=1.15,
        use_dynamic_shifting=True,
    ):
        # 使用正确的参数调用父类
        super().__init__(
            num_train_timesteps=num_train_timesteps,
            shift=shift,
            use_dynamic_shifting=use_dynamic_shifting,
            base_shift=base_shift,
            max_shift=max_shift,
            base_image_seq_len=base_image_seq_len,
            max_image_seq_len=max_image_seq_len,
            shift_terminal=shift_terminal,
            invert_sigmas=False,
            use_karras_sigmas=False,
            use_exponential_sigmas=False,
            use_beta_sigmas=False,
        )
        
        # 保存原有的自定义参数
        self.sigma_max = sigma_max
        self.sigma_min = sigma_min
        self.inverse_timesteps = inverse_timesteps
        self.extra_one_step = extra_one_step
        self.reverse_sigmas = reverse_sigmas
        self.exponential_shift = exponential_shift
        self.exponential_shift_mu = exponential_shift_mu
        
        # 初始化权重
        with torch.no_grad():
            num_timesteps = num_train_timesteps
            x = torch.arange(num_timesteps, dtype=torch.float32)
            y = torch.exp(-2 * ((x - num_timesteps / 2) / num_timesteps) ** 2)
            y_shifted = y - y.min()
            bsmntw_weighing = y_shifted * (num_timesteps / y_shifted.sum())
            
            timesteps = torch.linspace(num_train_timesteps, 1, num_timesteps, device='cpu')
            self.linear_timesteps = timesteps
            self.linear_timesteps_weights = bsmntw_weighing
            
        # 初始化时间步
        self.set_timesteps(num_inference_steps)
        self.training = False

    def set_timesteps(self, num_inference_steps=100, denoising_strength=1.0, training=False, shift=None, dynamic_shift_len=None):
        if shift is not None:
            self._shift = shift  # 更新内部shift值
            
        device = torch.device('cpu')  # 默认CPU，后续会根据输入自动转换
        
        sigma_start = self.sigma_min + (self.sigma_max - self.sigma_min) * denoising_strength
        
        if self.extra_one_step:
            self.sigmas = torch.linspace(sigma_start, self.sigma_min, num_inference_steps + 1)[:-1]
        else:
            self.sigmas = torch.linspace(sigma_start, self.sigma_min, num_inference_steps)
            
        if self.inverse_timesteps:
            self.sigmas = torch.flip(self.sigmas, dims=[0])
            
        if self.exponential_shift:
            mu = self.calculate_shift(dynamic_shift_len) if dynamic_shift_len is not None else self.exponential_shift_mu
            self.sigmas = math.exp(mu) / (math.exp(mu) + (1 / self.sigmas - 1))
        else:
            self.sigmas = self.shift * self.sigmas / (1 + (self.shift - 1) * self.sigmas)
            
        if self.shift_terminal is not None:
            one_minus_z = 1 - self.sigmas
            scale_factor = one_minus_z[-1] / (1 - self.shift_terminal)
            self.sigmas = 1 - (one_minus_z / scale_factor)
            
        if self.reverse_sigmas:
            self.sigmas = 1 - self.sigmas
            
        self.timesteps = self.sigmas * self.num_train_timesteps
        
        if training:
            x = self.timesteps
            y = torch.exp(-2 * ((x - num_inference_steps / 2) / num_inference_steps) ** 2)
            y_shifted = y - y.min()
            bsmntw_weighing = y_shifted * (num_inference_steps / y_shifted.sum())
            self.linear_timesteps_weights = bsmntw_weighing
            self.training = True
        else:
            self.training = False

    def step(self, model_output, timestep, sample, to_final=False, **kwargs):
        """
        Perform one reverse diffusion step.
        Args:
            model_output: predicted noise or x0, shape (B,C,H,W)
            timestep: int, scalar tensor, or (B,) tensor
            sample: current noised sample, shape (B,C,H,W)
            to_final: whether this is the final step
        """
        device = sample.device
        timesteps = self.timesteps.to(device)
        sigmas = self.sigmas.to(device)
        timestep = timestep.to(device).to(torch.float32)
        sample = sample.to(torch.float32)

        # 处理批量timestep
        if timestep.dim() == 0:
            timestep = timestep.unsqueeze(0)
            
        timestep_id = torch.argmin((timesteps[:, None] - timestep[None, :]).abs(), dim=0)
        sigma = sigmas[timestep_id]

        if to_final or (timestep_id.max().item() + 1 >= len(timesteps)):
            sigma_ = torch.tensor(
                1 if (self.inverse_timesteps or self.reverse_sigmas) else 0,
                device=device,
                dtype=sigma.dtype,
            )
            if sigma.dim() > 0:
                sigma_ = sigma_.expand_as(sigma)
        else:
            sigma_ = sigmas[timestep_id + 1]

        # 广播 sigma
        while sigma.ndim < sample.ndim:
            sigma = sigma.unsqueeze(-1)
            sigma_ = sigma_.unsqueeze(-1)

        prev_sample = sample + model_output * (sigma_ - sigma)
        return prev_sample.to(dtype=sample.dtype)
    
    def getsigmas(self, timestep, device, ndim: Optional[int] = None):
        """
        Get sigma values for given timesteps.
        Args:
            timestep: int, scalar tensor, or (B,) tensor
        """
        timesteps = self.timesteps.to(device)
        sigmas = self.sigmas.to(device)
        timestep = timestep.to(device)
        if timestep.dim() == 0:
            timestep = timestep.unsqueeze(0)

        timestep_id = torch.argmin((timesteps[:, None] - timestep[None, :]).abs(), dim=0)
        sigma = sigmas[timestep_id]
        if ndim is not None:
            while sigma.ndim < ndim:
                sigma = sigma.unsqueeze(-1)
        return sigma
    def return_to_timestep(self, timestep, sample, sample_stablized):
        """
        Reconstruct model output (predicted noise or x0) from given sample.
        """
        device = sample.device
        timesteps = self.timesteps.to(device)
        sigmas = self.sigmas.to(device)
        
        if timestep.dim() == 0:
            timestep = timestep.unsqueeze(0)

        timestep_id = torch.argmin((timesteps[:, None] - timestep[None, :]).abs(), dim=0)
        sigma = sigmas[timestep_id]

        # 广播 sigma
        while sigma.ndim < sample.ndim:
            sigma = sigma.unsqueeze(-1)

        model_output = (sample - sample_stablized) / sigma
        return model_output

    def add_noise(self, original_samples, noise, timestep):
        """
        Add noise to clean samples at given timesteps.
        Args:
            original_samples: clean data, shape (B,C,H,W)
            noise: Gaussian noise, same shape as original_samples
            timestep: int, scalar tensor, or (B,) tensor
        """
        device = original_samples.device
        timesteps = self.timesteps.to(device)
        sigmas = self.sigmas.to(device)
        
        if timestep.dim() == 0:
            timestep = timestep.unsqueeze(0)

        timestep_id = torch.argmin((timesteps[:, None] - timestep[None, :]).abs(), dim=0)
        sigma = sigmas[timestep_id]

        # 广播 sigma
        while sigma.ndim < original_samples.ndim:
            sigma = sigma.unsqueeze(-1)

        sample = (1 - sigma) * original_samples + sigma * noise
        return sample

    def training_target(self, sample, noise, timestep):
        target = noise - sample
        return target

    def training_weight(self, timestep):
        timestep_id = torch.argmin((self.timesteps[:, None] - timestep[None, :].to(self.timesteps.device)).abs(), dim=0)
        weights = self.linear_timesteps_weights[timestep_id]
        return weights.to(timestep.device)

    def calculate_shift(
        self,
        image_seq_len,
        base_seq_len: int = 256,
        max_seq_len: int = 4096,
        base_shift: float = 0.5,
        max_shift: float = 0.9,
    ):
        m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
        b = base_shift - m * base_seq_len
        mu = image_seq_len * m + b
        return mu

    @property
    def num_train_timesteps(self):
        """获取训练时间步数"""
        return self.config.num_train_timesteps

    @property
    def shift_terminal(self):
        """获取shift_terminal值"""
        return getattr(self.config, 'shift_terminal', None)