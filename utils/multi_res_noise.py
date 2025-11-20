import math
import torch


def multi_res_noise_like(
    x, strength=0.9, downscale_strategy="original", generator=None, device=None
):
    """
    Generate multi-resolution noise for improved training stability.
    
    Args:
        x: Input tensor to match shape
        strength: Strength of multi-resolution noise (default: 0.9)
        downscale_strategy: Strategy for downscaling ("original")
        generator: Random number generator
        device: Device for computation
    
    Returns:
        Multi-resolution noise tensor
    """
    if torch.is_tensor(strength):
        strength = strength.reshape((-1, 1, 1, 1))
    b, c, w, h = x.shape

    if device is None:
        device = x.device

    up_sampler = torch.nn.Upsample(size=(w, h), mode="bilinear")
    noise = torch.randn(x.shape, device=x.device, generator=generator)

    if "original" == downscale_strategy:
        for i in range(10):
            r = (
                torch.rand(1, generator=generator, device=device) * 2 + 2
            )  # Rather than always going 2x,
            w, h = max(1, int(w / (r**i))), max(1, int(h / (r**i)))
            noise += (
                up_sampler(
                    torch.randn(b, c, w, h, generator=generator, device=device).to(x)
                )
                * strength**i
            )
            if w == 1 or h == 1:
                break  # Lowest resolution is 1x1

    noise = noise / noise.std()  # Scaled back to roughly unit variance
    return noise
