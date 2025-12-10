import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import sys
from pathlib import Path

current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

from typing import List, Optional, Tuple

import torch
import numpy as np
from PIL import Image
import gradio as gr
import matplotlib.pyplot as plt
from scipy.ndimage import binary_dilation

# === 引入 ImageSlider ===
try:
    from gradio_imageslider import ImageSlider
except ImportError:
    print("⚠️ Warning: gradio_imageslider not installed. Using standard Image component fallback.")
    ImageSlider = None 

from pipelines.flux_image_new import FluxImagePipeline
from models.utils import load_state_dict
from trainers.unified_dataset import UnifiedDataset, gen_points
from utils.eval_multiple_datasets import parse_flux_model_configs
from models.flux_dit import FluxDiTStateDictConverter
converter = FluxDiTStateDictConverter()

# 全局变量
pipe = None
current_model = None
MODEL_INPUT_SIZE = 768
DISPLAY_LONG_SIDE = 768
resolution = MODEL_INPUT_SIZE
torch_dtype = torch.bfloat16

### Please Change the model root path below to your own model directory
model_root = "/mnt/nfs/share_model/FLUX.1-Kontext-dev"

# 模型配置
MODEL_CONFIGS = {
    "Depth_Lora": {
        "path": "ckpts/depth_lora.safetensors",
        "task": "depth"
    },
    "Normal_Lora": {
        "path": "ckpts/normal_lora.safetensors",
        "task": "normal"
    },
    "Matting_Lora": {
        "path": "ckpts/matting_lora.safetensors",
        "task": "matting"
    },
    "Depth_Full": {
        "path": "ckpts/depth.safetensors",
        "task": "depth"
    },
    "Normal_Full": {
        "path": "ckpts/normal.safetensors",
        "task": "normal"
    },
    "Matting_Full": {
        "path": "ckpts/matting.safetensors",
        "task": "matting"
    },
}

# 全局变量存储
selected_points = []
original_image = None
brush_mask = None 

# ================= 工具函数 =================

def resize_image_to_square(image: Image.Image, target_size: int = MODEL_INPUT_SIZE) -> Image.Image:
    if image.width == target_size and image.height == target_size:
        return image
    return image.resize((target_size, target_size), Image.Resampling.LANCZOS)

def resize_long_side(image: Image.Image, target_long_side: int = DISPLAY_LONG_SIDE) -> Image.Image:
    width, height = image.size
    long_side = max(width, height)
    if long_side <= target_long_side:
        return image
    scale = target_long_side / long_side
    new_width = max(1, int(width * scale))
    new_height = max(1, int(height * scale))
    return image.resize((new_width, new_height), Image.Resampling.LANCZOS)

def resize_array_long_side(image_array: np.ndarray, target_long_side: int = DISPLAY_LONG_SIDE) -> np.ndarray:
    h, w = image_array.shape[:2]
    if max(h, w) <= target_long_side:
        return image_array
    scale = target_long_side / max(h, w)
    new_h = max(1, int(h * scale))
    new_w = max(1, int(w * scale))
    try:
        import cv2
        return cv2.resize(image_array, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    except ImportError:
        pil_image = Image.fromarray(image_array)
        resized = pil_image.resize((new_w, new_h), Image.Resampling.NEAREST)
        return np.array(resized)

# ================= 初始化与模型加载 =================

def initialize_pipeline():
    global pipe
    if pipe is not None:
        return
    print("Loading FluxImagePipeline ...")
    pipe = FluxImagePipeline.from_pretrained(
        torch_dtype=torch_dtype,
        device="cuda",
        model_configs=parse_flux_model_configs(model_root)
    )
    print("Pipeline loaded successfully!")

def load_model(model_name: str, progress=gr.Progress()):
    global current_model
    if model_name == current_model:
        return f"✓ MODEL ACTIVE | {model_name.upper()} ALREADY LOADED"
    
    if pipe is None:
        progress(0, desc="⚡ Initializing Neural Pipeline...")
        initialize_pipeline()
    
    model_config = MODEL_CONFIGS[model_name]
    state_dict_path = model_config["path"]
    
    progress(0.0, desc=f"📡 Loading {model_name.upper()} Weights...")
    if "lora" in state_dict_path:
        state_dict = load_state_dict(os.path.join(model_root, "flux1-kontext-dev.safetensors"))
        pipe.dit.load_state_dict(converter.from_civitai(state_dict))
        pipe.load_lora(pipe.dit, state_dict_path, hotload=False)
    else:
        state_dict = load_state_dict(os.path.join(model_root, "flux1-kontext-dev.safetensors"))
        pipe.dit.load_state_dict(converter.from_civitai(state_dict))
        state_dict = load_state_dict(state_dict_path)
        pipe.dit.load_state_dict(state_dict)
    current_model = model_name
    
    progress(1.0, desc="✓ Model Loaded!")
    return f"✓ NEURAL MODEL LOADED | {model_name.upper()} READY FOR INFERENCE"

def handle_model_switch(model_name: str):
    return load_model(model_name)

# ================= 图像处理逻辑 =================

def create_alpha_mask_from_points_and_brush(width: int, height: int, 
                                             points: List[Tuple[int, int]] = None, 
                                             brush_mask_original: np.ndarray = None,
                                             orig_w: int = None, orig_h: int = None,
                                             point_radius: int = 100) -> np.ndarray:
    alpha = np.zeros((height, width), dtype=np.float32)
    
    if points and len(points) > 0:
        for point_x, point_y in points:
            scaled_x = int(point_x * width / orig_w)
            scaled_y = int(point_y * height / orig_h)
            y_coords, x_coords = np.ogrid[:height, :width]
            mask = (x_coords - scaled_x) ** 2 + (y_coords - scaled_y) ** 2 <= point_radius ** 2
            alpha[mask] = 1.0
    
    if brush_mask_original is not None:
        brush_mask_resized = Image.fromarray((brush_mask_original * 255).astype(np.uint8)).resize((width, height), Image.NEAREST)
        brush_mask_resized = np.array(brush_mask_resized) / 255.0
        alpha = np.maximum(alpha, brush_mask_resized)
    
    return alpha

def inference(model_name: str, image: np.ndarray, click_points: Optional[List[Tuple[int, int]]] = None, 
              num_inference_steps: int = 4, seed: int = 42) -> Tuple[Image.Image, str]:
    if image is None:
        return None, "⚠️ ERROR | NO IMAGE INPUT"
    
    load_model(model_name)
    model_config = MODEL_CONFIGS[model_name]
    task = model_config["task"]
    
    transform = UnifiedDataset.default_image_operator(height=resolution, width=resolution)
    
    orig_h, orig_w = image.shape[:2]
    pil_image = Image.fromarray(image)
    pil_image_sq = resize_image_to_square(pil_image, MODEL_INPUT_SIZE)
    
    try:
        out_np = None
        if task in ["depth", "normal"]:
            out_np = pipe(
                prompt=f"Transform to {task} map while maintaining original composition",
                kontext_images=transform(pil_image_sq),
                height=MODEL_INPUT_SIZE, width=MODEL_INPUT_SIZE,
                embedded_guidance=1,
                num_inference_steps=num_inference_steps,
                seed=seed,
                output_type="np",
                rand_device="cuda",
                task=task,
            )
            
            if task == "depth":
                if out_np.ndim == 3:
                    out_np = np.mean(out_np, axis=2)
                out_np = out_np + 1e-3
                out_np = np.pad(out_np, 20, mode='constant', constant_values=0)
                cmap = plt.get_cmap('Spectral')
                out_np = cmap(out_np)[:, :, :3]
                out_np = out_np[20:-20, 20:-20]
                out_np = (out_np * 255).astype(np.uint8)
            elif task == "normal":
                out_np = (out_np + 1) / 2 * 255.0
                out_np = out_np.astype(np.uint8)
                
        elif task == "matting":
            alpha = create_alpha_mask_from_points_and_brush(
                resolution, resolution,
                points=click_points,
                brush_mask_original=brush_mask,
                orig_w=orig_w, orig_h=orig_h,
                point_radius=100
            )

            points, _ = gen_points(alpha, num_points=10, radius=30)
            points_tensor = torch.from_numpy(points * 2 - 1).repeat(3, 1, 1).to("cuda")
            kontext_inputs = [transform(pil_image_sq), points_tensor]

            out_np = pipe(
                prompt=f"Transform to {task} map while maintaining original composition",
                kontext_images=kontext_inputs,
                height=MODEL_INPUT_SIZE, width=MODEL_INPUT_SIZE,
                embedded_guidance=1,
                num_inference_steps=num_inference_steps,
                seed=seed,
                output_type="np",
                rand_device="cuda",
                task=task,
            )
            out_np = ((out_np) * 255.0).astype(np.uint8)
        
        out_pil = Image.fromarray(out_np)
        out_pil = out_pil.resize((orig_w, orig_h), Image.Resampling.NEAREST)
        out_pil = resize_long_side(out_pil, DISPLAY_LONG_SIDE)
        
        return out_pil, f"✓ INFERENCE COMPLETE | MODEL: {model_name.upper()} | STATUS: SUCCESS"
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return None, f"❌ INFERENCE FAILED | ERROR: {str(e)}"

def draw_points_on_image(image: np.ndarray, points: List[Tuple[int, int]], 
                         point_radius: int = 9, coverage_radius: int = 100, 
                         show_coverage: bool = True) -> np.ndarray:
    # 始终在原图的拷贝上绘制，避免叠加污染
    img_with_markers = image.copy().astype(np.float32)
    
    for x, y in points:
        if show_coverage:
            for dx in range(-coverage_radius, coverage_radius + 1):
                for dy in range(-coverage_radius, coverage_radius + 1):
                    if dx * dx + dy * dy <= coverage_radius * coverage_radius:
                        new_x, new_y = x + dx, y + dy
                        if 0 <= new_x < image.shape[1] and 0 <= new_y < image.shape[0]:
                            img_with_markers[new_y, new_x] = img_with_markers[new_y, new_x] * 0.7 + np.array([0, 255, 0]) * 0.3
        for dx in range(-point_radius, point_radius + 1):
            for dy in range(-point_radius, point_radius + 1):
                if dx * dx + dy * dy <= point_radius * point_radius:
                    new_x, new_y = x + dx, y + dy
                    if 0 <= new_x < image.shape[1] and 0 <= new_y < image.shape[0]:
                        img_with_markers[new_y, new_x] = [255, 0, 0]
    return img_with_markers.astype(np.uint8)

# ================= 事件处理 (修复重点) =================

def on_image_upload(image):
    """
    处理图片上传：
    1. 提取原图并保存到全局变量 original_image。
    2. 重置 selected_points。
    3. 关键修复：不要返回图片给 input_image，只返回状态和清空结果。
    """
    global selected_points, original_image, brush_mask
    selected_points = []
    brush_mask = None

    if image is None:
        original_image = None
        return "⚠️ ERROR | INVALID IMAGE FORMAT", None

    # ImageEditor 默认返回的是 dict
    if isinstance(image, dict):
        # 优先取 background，如果为空取 composite
        bg = image.get('background')
        if bg is None:
            bg = image.get('composite')
        
        if bg is None:
            original_image = None
            return "⚠️ ERROR | UNABLE TO READ IMAGE", None
        
        # 保存纯净原图 (去除任何alpha通道如果不需要，或者保留)
        if bg.ndim == 3 and bg.shape[2] == 4:
             original_image = bg[:, :, :3] # 只要RGB
        else:
             original_image = bg
    else:
        # 假如是直接 numpy
        original_image = image
    
    # ⚠️ 关键：这里只返回 Text 和 None(清空结果)，不返回 image
    return "✓ IMAGE LOADED", None

def on_image_click(image, evt: gr.SelectData):
    """
    处理点击打点。
    这里需要返回图片来显示红点。
    """
    global selected_points, original_image
    
    # 如果 original_image 还没初始化，尝试从当前的 image 参数恢复
    if original_image is None:
        if isinstance(image, dict):
            bg = image.get('background')
            if bg is not None:
                original_image = bg[:,:,:3] if bg.shape[2]==4 else bg
        elif isinstance(image, np.ndarray):
            original_image = image
            
    if original_image is None:
        return image, "⚠️ ERROR | NO BASE IMAGE FOUND"

    # 记录点坐标
    x, y = evt.index[0], evt.index[1]
    selected_points.append((x, y))
    
    # 计算半径
    orig_h, orig_w = original_image.shape[:2]
    display_coverage_radius = int(100 * orig_w / resolution)
    
    # 在 干净的 original_image 上重新绘制所有点
    # 这样可以避免多次点击导致圆圈叠加颜色变深或模糊
    img_with_markers = draw_points_on_image(
        original_image, 
        selected_points,
        point_radius=9,
        coverage_radius=display_coverage_radius,
        show_coverage=True
    )
    
    # 返回给 Editor 显示
    return img_with_markers, f"🎯 POINT SELECTED | TOTAL: {len(selected_points)} TARGET(S) MARKED"

def reset_selection(image):
    """
    重置：
    需要清空 Editor 的画笔痕迹和点，所以这里需要返回 clean_editor 给组件
    """
    global selected_points, original_image, brush_mask
    selected_points = []
    brush_mask = None
    
    clean_editor = None
    if original_image is not None:
        # 重构 Editor 需要的字典格式，清空 layers
        clean_editor = {
            'background': original_image,
            'layers': [],
            'composite': original_image
        }
    else:
        # 如果没有原图，就全空
        clean_editor = None
            
    return clean_editor, "🔄 WORKSPACE RESET | ALL SELECTIONS CLEARED", None

def run_inference(model_name, image, num_inference_steps, seed):
    global selected_points, original_image, brush_mask
    
    if original_image is None:
        return "⚠️ ERROR | NO SOURCE IMAGE", None

    model_config = MODEL_CONFIGS[model_name]
    task = model_config["task"]
    
    # 1. 提取画笔 Mask (仅Matting)
    if task == "matting":
        # 此时 image 参数是最新的 Editor 状态，包含了用户的涂抹层
        if isinstance(image, dict) and 'layers' in image and len(image['layers']) > 0:
            # 合并所有 layer (通常只有一个)
            # Gradio 的 layer 通常是 RGBA，其中 A 是涂抹的不透明度
            # 我们需要把所有有涂抹的地方提取出来
            mask_combined = np.zeros(original_image.shape[:2], dtype=np.float32)
            
            for layer in image['layers']:
                if layer is not None:
                    # layer 形状 (H, W, 4)
                    alpha = layer[:, :, 3] / 255.0
                    mask_combined = np.maximum(mask_combined, alpha)
            
            if np.max(mask_combined) > 0:
                # 膨胀一下 mask
                kernel_size = 40
                kernel = np.zeros((kernel_size*2+1, kernel_size*2+1))
                y, x = np.ogrid[-kernel_size:kernel_size+1, -kernel_size:kernel_size+1]
                mask_circle = x**2 + y**2 <= kernel_size**2
                kernel[mask_circle] = 1
                brush_mask = binary_dilation(mask_combined > 0, structure=kernel).astype(np.float32)
            else:
                brush_mask = None
        else:
            brush_mask = None
    
    # 3. 执行推理，使用全局 original_image 保证画质最清晰
    result_pil, message = inference(model_name, original_image, selected_points if selected_points else None, num_inference_steps, seed)

    if result_pil is None:
        return message, None
    
    # 4. 准备输出
    input_pil = Image.fromarray(original_image)
    input_pil_display = resize_long_side(input_pil, DISPLAY_LONG_SIDE)
    
    return message, (input_pil_display, result_pil)

# ================= 界面构建 =================
def create_gradio_interface():
    custom_theme = gr.themes.Base(
        primary_hue=gr.themes.colors.cyan,
        secondary_hue=gr.themes.colors.slate,
        neutral_hue=gr.themes.colors.slate,
        font=gr.themes.GoogleFont("Inter"),
    ).set(
        body_background_fill="*neutral_950",
        block_title_text_color="*neutral_100",
        button_primary_background_fill="linear-gradient(90deg, *primary_600, *secondary_600)",
        slider_color="*primary_500",
    )
    
    with gr.Blocks(title="Edit2Percieve AI", theme=custom_theme, css="""
        .gradio-container { max-width: 100% !important; background: #0a0a0a !important; }
        h1 { color: #64ffda; text-align: center; }
    """) as demo:
        gr.HTML("""<div style="text-align: center;"><h1>⚡ EDIT2PERCIEVE AI ⚡</h1></div>""")

        result_state = gr.State(value=None)

        with gr.Row():
            with gr.Column(scale=1):
                input_image = gr.ImageEditor(
                    label="🎨 INPUT WORKSPACE",
                    type="numpy",
                    brush=gr.Brush(colors=["#00FF00"], default_size=40),
                    eraser=gr.Eraser(default_size=40),
                    height=550,
                    sources=["upload", "clipboard"]
                )
                
                with gr.Row():
                    reset_btn = gr.Button("🔄 RESET", size="sm", variant="secondary")
                    run_btn = gr.Button("⚡ EXECUTE", variant="primary", size="sm")
                
                with gr.Accordion("⚙️ NEURAL CONFIG", open=True):
                    model_dropdown = gr.Dropdown(choices=list(MODEL_CONFIGS.keys()), value="Depth_Lora", label="🎯 TASK MODE")
                    with gr.Row():
                        num_steps = gr.Slider(1, 10, value=1, step=1, label="🔢 INFERENCE STEPS")
                        seed = gr.Number(value=42, label="🎲 RANDOM SEED", precision=0)
            
            with gr.Column(scale=1):
                @gr.render(inputs=result_state)
                def show_output(result_data):
                    if result_data is None:
                        gr.Image(label="📊 OUTPUT ANALYSIS", interactive=False, height=550, value=None)
                    else:
                        if ImageSlider:
                            ImageSlider(value=result_data, label="🔍 COMPARISON VIEW", type="pil", position=0.5, height=550)
                        else:
                            gr.Image(value=result_data[1], label="📊 OUTPUT ANALYSIS", height=550)
                
                status_text = gr.Textbox(label="💻 SYSTEM STATUS", interactive=False, value="🟢 READY")

        # --- 事件绑定修复 ---

        # 1. 上传图片: 修改 outputs，移除 input_image，防止死循环
        input_image.upload(
            on_image_upload,
            inputs=[input_image],
            outputs=[status_text, result_state]  # ❌ 移除了 input_image
        )
        
        # 2. 点击打点: 需要更新 input_image 以显示红点，这是安全的，因为不是 upload 事件
        input_image.select(
            on_image_click,
            inputs=[input_image],
            outputs=[input_image, status_text]
        )
        
        # 3. 重置: 显式更新 input_image 以清空
        reset_btn.click(
            reset_selection,
            inputs=[input_image],
            outputs=[input_image, status_text, result_state]
        )
        
        model_dropdown.change(handle_model_switch, inputs=[model_dropdown], outputs=[status_text])
        
        run_btn.click(
            run_inference,
            inputs=[model_dropdown, input_image, num_steps, seed],
            outputs=[status_text, result_state]
        )
    
    return demo

if __name__ == "__main__":
    initialize_pipeline()
    demo = create_gradio_interface()
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)