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

# 全局变量
pipe = None
current_model = None
MODEL_INPUT_SIZE = 512
DISPLAY_LONG_SIDE = 768
resolution = MODEL_INPUT_SIZE
torch_dtype = torch.bfloat16

### Please Change the model root path below to your own model directory
model_root = "/mnt/nfs/share_model/FLUX.1-Kontext-dev"



# 模型配置
MODEL_CONFIGS = {
    "Depth": {
        "path": "ckpts/edit2percieve_depth.safetensors",
        "task": "depth"
    },
    "Normal": {
        "path": "ckpts/edit2percieve_normal.safetensors",
        "task": "normal"
    },
    "Matting": {
        "path": "ckpts/edit2percieve_matting.safetensors",
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
    return image.resize((target_size, target_size), Image.Resampling.NEAREST)

def resize_long_side(image: Image.Image, target_long_side: int = DISPLAY_LONG_SIDE) -> Image.Image:
    width, height = image.size
    long_side = max(width, height)
    if long_side <= target_long_side:
        return image
    scale = target_long_side / long_side
    new_width = max(1, int(width * scale))
    new_height = max(1, int(height * scale))
    return image.resize((new_width, new_height), Image.Resampling.NEAREST)

def resize_array_long_side(image_array: np.ndarray, target_long_side: int = DISPLAY_LONG_SIDE) -> np.ndarray:
    h, w = image_array.shape[:2]
    
    # 如果图片已经小于等于目标尺寸，直接返回原数组（零拷贝）
    if max(h, w) <= target_long_side:
        return image_array
    
    # 使用cv2进行最快速的resize（如果可用），否则用PIL NEAREST
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
    
    progress(0.5, desc=f"📡 Loading {model_name.upper()} Weights...")
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

# ================= 事件处理 =================

def on_image_upload(image):
    """处理图片上传：重置全局状态，并清空右侧滑块"""
    global selected_points, original_image, brush_mask
    selected_points = []
    brush_mask = None

    if image is None:
        original_image = None
        return None, "⚠️ ERROR | INVALID IMAGE FORMAT", None

    # 处理 ImageEditor 字典输入
    if isinstance(image, dict):
        bg = image.get('background')
        if bg is None:
            bg = image.get('composite')
        if bg is None:
            original_image = None
            return image, "⚠️ ERROR | UNABLE TO READ IMAGE", None
        
        # 直接使用原图，不做任何处理
        original_image = bg
        
        return image, "✓ IMAGE LOADED", None
    else:
        # 处理直接 numpy 输入（防止万一）
        original_image = image
        return image, "✓ IMAGE LOADED", None

def on_image_click(image, evt: gr.SelectData):
    """处理图片点击：只更新左侧 Editor，不影响右侧"""
    global selected_points, original_image
    
    if image is None:
        return image, "⚠️ ERROR | NO IMAGE DETECTED"
    
    # 确保 original_image 存在
    if original_image is None:
        if isinstance(image, dict):
            bg = image.get('background')
            if bg is not None:
                original_image = bg.copy()
            else:
                return image, "⚠️ ERROR | INVALID BACKGROUND DATA"
        else:
            original_image = image.copy()
    
    x, y = evt.index[0], evt.index[1]
    selected_points.append((x, y))
    
    # 计算显示半径
    orig_h, orig_w = original_image.shape[:2]
    display_coverage_radius = int(100 * orig_w / resolution)
    
    # 在原图上绘制点
    img_with_markers = draw_points_on_image(
        original_image, 
        selected_points,
        point_radius=9,
        coverage_radius=display_coverage_radius,
        show_coverage=True
    )
    
    # 返回：(Editor更新, 状态文字)
    return img_with_markers, f"🎯 POINT SELECTED | TOTAL: {len(selected_points)} TARGET(S) MARKED"

def reset_selection(image):
    """重置：清空点和mask，恢复左侧原图，清空右侧结果"""
    global selected_points, original_image, brush_mask
    selected_points = []
    brush_mask = None
    
    # 构造干净的 Editor 数据
    clean_editor = None
    if original_image is not None:
        if isinstance(image, dict):
            clean_editor = {
                'background': original_image,
                'layers': [],
                'composite': original_image
            }
        else:
            clean_editor = original_image
            
    return clean_editor, "🔄 WORKSPACE RESET | ALL SELECTIONS CLEARED", None

def run_inference(model_name, image, num_inference_steps, seed):
    """
    执行推理。
    关键修改：此函数不再返回 input_image，避免触发 Editor 的刷新事件导致死循环。
    """
    global selected_points, original_image, brush_mask
    
    model_config = MODEL_CONFIGS[model_name]
    task = model_config["task"]
    
    # 1. 提取画笔 Mask (仅Matting)
    if task == "matting":
        if isinstance(image, dict) and 'layers' in image and len(image['layers']) > 0:
            mask_layer = image['layers'][0]
            if mask_layer is not None and len(mask_layer.shape) >= 3:
                if mask_layer.shape[-1] == 4:
                    brush_alpha = mask_layer[:, :, 3] / 255.0
                    kernel_size = 40
                    kernel = np.zeros((kernel_size*2+1, kernel_size*2+1))
                    y, x = np.ogrid[-kernel_size:kernel_size+1, -kernel_size:kernel_size+1]
                    mask_circle = x**2 + y**2 <= kernel_size**2
                    kernel[mask_circle] = 1
                    brush_mask = binary_dilation(brush_alpha > 0, structure=kernel).astype(np.float32)
                else:
                    brush_mask = None
        else:
            brush_mask = None
    
    # 2. 确定输入图像
    if original_image is not None:
        input_image = original_image
    elif isinstance(image, dict):
        bg = image.get('background')
        input_image = bg if bg is not None else image.get('composite')
    else:
        input_image = image
    
    if input_image is None:
        # 返回：状态文字, Slider不变(None)
        return "⚠️ ERROR | NO INPUT IMAGE DETECTED", None
    
    # 3. 执行推理
    result_pil, message = inference(model_name, input_image, selected_points if selected_points else None, num_inference_steps, seed)

    if result_pil is None:
        return message, None
    
    # 4. 准备 Slider 数据（延迟resize到显示时）
    input_pil = Image.fromarray(input_image)
    input_pil_display = resize_long_side(input_pil, DISPLAY_LONG_SIDE)
    
    # 返回：(状态文字, (原图, 结果图))
    return message, (input_pil_display, result_pil)

# ================= 界面构建 =================
def create_gradio_interface():
    # 科技风黑色主题配置
    custom_theme = gr.themes.Base(
        primary_hue=gr.themes.colors.cyan,
        secondary_hue=gr.themes.colors.slate,
        neutral_hue=gr.themes.colors.slate,
        font=gr.themes.GoogleFont("Inter"),
    ).set(
        body_background_fill="*neutral_950",
        body_background_fill_dark="*neutral_950",
        background_fill_primary="*neutral_900",
        background_fill_primary_dark="*neutral_900",
        background_fill_secondary="*neutral_800",
        background_fill_secondary_dark="*neutral_800",
        border_color_primary="*neutral_700",
        border_color_primary_dark="*neutral_700",
        color_accent="*primary_500",
        color_accent_soft="*primary_400",
        block_title_text_color="*neutral_100",
        block_label_text_color="*neutral_200",
        body_text_color="*neutral_200",
        body_text_color_subdued="*neutral_400",
        button_primary_background_fill="linear-gradient(90deg, *primary_600, *secondary_600)",
        button_primary_background_fill_hover="linear-gradient(90deg, *primary_500, *secondary_500)",
        button_primary_text_color="white",
        button_secondary_background_fill="*neutral_700",
        button_secondary_background_fill_hover="*neutral_600",
        button_secondary_text_color="*neutral_100",
        input_background_fill="*neutral_800",
        slider_color="*primary_500",
    )
    
    with gr.Blocks(title="Edit2Percieve AI", theme=custom_theme, css="""
        .gradio-container {
            max-width: 100% !important;
            background: linear-gradient(135deg, #0a0a0a 0%, #1a1a2e 100%) !important;
        }
        .contain {
            background: rgba(20, 20, 30, 0.8) !important;
            border: 1px solid rgba(100, 255, 218, 0.2) !important;
            border-radius: 12px !important;
            backdrop-filter: blur(10px) !important;
        }
        h1 {
            background: linear-gradient(90deg, #00d4ff, #00ffa3);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-weight: 700;
            font-size: 2.5em !important;
            text-align: center;
            margin-bottom: 0.5em;
            text-shadow: 0 0 30px rgba(0, 212, 255, 0.3);
        }
        .subtitle {
            text-align: center;
            color: #64ffda;
            font-size: 1.1em;
            margin-bottom: 2em;
            font-weight: 300;
            letter-spacing: 1px;
        }
        button {
            border-radius: 8px !important;
            font-weight: 600 !important;
            transition: all 0.3s ease !important;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        button:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 20px rgba(0, 212, 255, 0.4) !important;
        }
        .primary {
            background: linear-gradient(90deg, #00d4ff, #00ffa3) !important;
            border: none !important;
        }
        label {
            color: #64ffda !important;
            font-weight: 600 !important;
            font-size: 0.95em !important;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
    """) as demo:
        gr.HTML("""
            <div style="text-align: center; margin-bottom: 20px;">
                <h1>⚡ EDIT2PERCIEVE AI ⚡</h1>
                <p class="subtitle">🔬 Advanced Multi-Task Visual Intelligence System 🔬</p>
                <p style="color: #888; font-size: 0.9em;">┃ UPLOAD ┃ ANALYZE ┃ TRANSFORM ┃</p>
            </div>
        """)

        # === 1. 定义一个 State 变量来存储推理结果 ===
        # 初始值为 None，类型为 Tuple[Image, Image]
        result_state = gr.State(value=None)

        with gr.Row(equal_height=True):
            # 左侧：输入
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
                    model_dropdown = gr.Dropdown(
                        choices=list(MODEL_CONFIGS.keys()),
                        value="Depth",
                        label="🎯 TASK MODE"
                    )
                    with gr.Row():
                        num_steps = gr.Slider(1, 10, value=4, step=1, label="🔢 INFERENCE STEPS")
                        seed = gr.Number(value=42, label="🎲 RANDOM SEED", precision=0)
            
            # 右侧：输出 (使用 @gr.render 动态生成)
            with gr.Column(scale=1):
                # 定义一个容器，内容将由 result_state 决定
                @gr.render(inputs=result_state)
                def show_output(result_data):
                    # 每次 result_state 变化，这个函数都会重新运行
                    # 从而创建全新的组件实例
                    if result_data is None:
                        # 状态为空时显示占位符
                        gr.Image(
                            label="📊 OUTPUT ANALYSIS", 
                            interactive=False, 
                            height=550, 
                            value=None, 
                            type="pil"
                        )
                    else:
                        # 状态有值时，创建全新的 ImageSlider
                        # 因为是新创建的，position=0.5 一定会生效
                        if ImageSlider:
                            ImageSlider(
                                value=result_data,
                                label="🔍 COMPARISON VIEW",
                                type="pil",
                                position=0.5,  # 强制居中
                                height=550
                            )
                        else:
                            gr.Image(value=result_data[1], label="📊 OUTPUT ANALYSIS", height=550)
                
                status_text = gr.Textbox(label="💻 SYSTEM STATUS", interactive=False, value="🟢 READY | AWAITING INPUT")

        # --- 事件绑定 ---
        
        demo.load(lambda: load_model("Depth"), outputs=[status_text])

        # 上传图片 -> 清空 result_state (变为None)
        input_image.upload(
            on_image_upload,
            inputs=[input_image],
            outputs=[input_image, status_text, result_state]
        )
        
        # 点击 -> 只更新左侧
        input_image.select(
            on_image_click,
            inputs=[input_image],
            outputs=[input_image, status_text]
        )
        
        # 重置 -> 清空 result_state
        reset_btn.click(
            reset_selection,
            inputs=[input_image],
            outputs=[input_image, status_text, result_state]
        )
        
        model_dropdown.change(
            handle_model_switch,
            inputs=[model_dropdown],
            outputs=[status_text]
        )
        
        # 运行推理 -> 更新 result_state
        # 当 result_state 更新时，右侧的 @gr.render 会自动触发
        run_btn.click(
            run_inference,
            inputs=[model_dropdown, input_image, num_steps, seed],
            outputs=[status_text, result_state]
        )
        
        # 底部彩蛋
        gr.HTML("""
            <div style="text-align: center; margin-top: 30px; padding: 20px; border-top: 1px solid rgba(100, 255, 218, 0.2);">
                <p style="color: #64ffda; font-size: 1.1em; font-weight: 500; letter-spacing: 2px;">
                    Present by 🥥🍉
                </p>
            </div>
        """)
    
    return demo
if __name__ == "__main__":
    initialize_pipeline() # 可以选择在这里初始化，或者懒加载
    demo = create_gradio_interface()
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)# 小施宝宝💗💗小施宝宝