# Edit2Percieve AI

⚡ Advanced Multi-Task Visual Intelligence System for Depth Estimation, Normal Map Generation, and Image Matting.

## 🎯 Features

- **Depth Estimation**: Generate high-quality depth maps from RGB images
- **Normal Eeneration**: Extract surface normal information from images
- **Interactive Matting**: Intelligent foreground/background separation
- **Interactive UI**: Beautiful Gradio-based web interface with real-time visualization
- **CLI Support**: Command-line interface for batch processing

## 📋 Requirements (Recommend)

- Python 3.12
- CUDA-capable GPU (recommended)
- 40GB+ VRAM for optimal performance

## 🚀 Installation

1. **Clone the repository**
```bash
git clone <your-repo-url>
cd open_source_infer
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Download FLUX.1-Kontext Model**

Download the FLUX.1-Kontext-dev model and place it in your desired directory:
```
/path/to/FLUX.1-Kontext-dev/
```

4. **Download Task-Specific Models**

Download our pre-trained models and place them in the `ckpts/` directory:
```
ckpts/
├── edit2percieve_depth.safetensors
├── edit2percieve_normal.safetensors
└── edit2percieve_matting.safetensors
```

## 💻 Usage

### Option 1: Web Interface (Recommended)

Launch the interactive Gradio UI:

```bash
python app.py
```

**Configuration:**
- Edit the `model_root` path in `app.py` (line 37) to point to your FLUX.1-Kontext model directory
- Open your browser and navigate to `http://localhost:7860`
- Upload an image, select a task (Depth/Normal/Matting), and click Execute

**Features:**
- 🎨 Interactive image editor with brush/eraser tools
- 🔍 Side-by-side comparison slider
- ⚙️ Adjustable inference parameters
- 🖱️ Point-based annotation for matting tasks

### Option 2: Command Line Interface

Run inference without GUI:

```bash
python inference.py
```

**Configuration:**
Edit the `__main__` section in `inference.py`:

```python
if __name__ == "__main__":
    # Set your model root path
    model_root = "/path/to/FLUX.1-Kontext-dev"
    
    inference(
        model_root=model_root,
        task="depth",  # Options: "depth", "normal", "matting"
        input_paths="samples/cat.jpg"  # Single image or comma-separated paths
    )
```

**Parameters:**
- `model_root`: Path to FLUX.1-Kontext model directory
- `task`: Task type - `"depth"`, `"normal"`, or `"matting"`
- `input_paths`: Input image path(s)
- `resolution`: Processing resolution (default: 768)
- `num_inference_steps`: Number of diffusion steps (default: 8)
- `seed`: Random seed for reproducibility (default: 42)
- `output_path`: Custom output path (optional)

## 📁 Project Structure

```
open_source_infer/
├── app.py                  # Gradio web interface
├── inference.py            # CLI inference script
├── requirements.txt        # Python dependencies
├── ckpts/                  # Model checkpoints directory
│   ├── edit2percieve_depth.safetensors
│   ├── edit2percieve_normal.safetensors
│   └── edit2percieve_matting.safetensors
├── samples/                # Sample images
├── pipelines/              # Inference pipelines
├── models/                 # Model architectures
├── trainers/               # Training utilities
└── utils/                  # Helper functions
```

## 🎨 Examples

### Depth Estimation
```python
inference(
    model_root="/path/to/FLUX.1-Kontext-dev",
    task="depth",
    input_paths="samples/cat.jpg"
)
```

### Normal Map Generation
```python
inference(
    model_root="/path/to/FLUX.1-Kontext-dev",
    task="normal",
    input_paths="samples/dog.jpg"
)
```

### Image Matting
```python
inference(
    model_root="/path/to/FLUX.1-Kontext-dev",
    task="matting",
    input_paths="samples/cat.jpg"
)
```

## ⚙️ Model Configuration

The models are configured in `MODEL_CONFIGS` dictionary:

```python
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
```

## 🙏 Acknowledgments

This project is built upon the FLUX.1-Kontext model architecture.

---

**Present by 🥥🍉**
