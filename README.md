
# Edit2Perceive: Image Editing Diffusion Models Are Strong Dense Perceivers
*Yiqing Shi, Yiren Song, Mike Zheng Shou*


[![arXiv](https://img.shields.io/badge/arXiv-2511.18673-b31b1b.svg)](https://arxiv.org/abs/2511.18673)
[![HuggingFace](https://img.shields.io/badge/Hugging%20Face-model-yellow.svg?logo=huggingface)](https://hf-mirror.com/Seq2Tri/Edit2Perceive/tree/main)

![Teaser](samples/teaser.png   )



## Installation

1. **Clone the repository**

    ```bash
    git clone https://github.com/showlab/Edit2Perceive.git
    cd Edit2Perceive
    conda create -n e2p python=3.12
    conda activate e2p
    pip install -r requirements.txt
    ```

2. **Download Base Model**

    Download the [FLUX.1-Kontext-dev](https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev) model:
    ```bash
    export HF_ENDPOINT=https://hf-mirror.com # if huggingface is not available, use this mirror
    hf download black-forest-labs/FLUX.1-Kontext-dev --exclude "transformer/" --local-dir ./FLUX.1-Kontext-dev
    ```
3. **Download Our Models**

    Download our pre-trained models and place them in the `ckpts/` directory. You can either download lora version (small size for fast validation) or full version (best quality but file is large):
    
    **Option1** Download LoRA weights
    ```bash
    hf download Seq2Tri/Edit2Perceive --local-dir ckpts/ --include "*lora.safetensors"
    ```
    **Option2** Download full model weights
    ```bash
    hf download Seq2Tri/Edit2Perceive --local-dir ckpts/ --exclude "*lora.safetensors"
    ```
    The Final Folder Sturcture should be like this:
    ```bash
    ckpts/
    ├── depth.safetensors
    ├── depth_lora.safetensors
    ├── normal.safetensors
    ├── normal_lora.safetensors
    ├── matting.safetensors
    └── matting_lora.safetensors
    ```

## Quick Start
### UI

```bash
python app.py
```
and then visit `http://localhost:7860`

### No UI

```bash
python inference.py
```



## Cite
If you find our work useful in your research please consider citing our paper:

```Bibtex
@misc{edit2perceive,
      title={Edit2Perceive: Image Editing Diffusion Models Are Strong Dense Perceivers}, 
      author={Yiqing Shi and Yiren Song and Mike Zheng Shou},
      year={2025},
      eprint={2511.18673},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2511.18673}, 
}
```

## Contact
If you have any questions, please feel free to contact yqshi@stu.pku.edu.cn