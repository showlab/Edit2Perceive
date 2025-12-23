<!-- here are the parameter explanation:
    --dataset_base_path : absolute path of datasets
    --dataset_metadata_path : metadata containing the reletive path of training files, columns are readed by `data_file_keys`
    --data_file_keys : the column names in dataset_metadata_path file, such as `kontext_images, image`
    --max_pixels "1048577" : max pixels of input image, not used
    --dataset_repeat "1" : dataset repeat, which is useful for small size dataset,
    --model_paths : the path of model (can be either FLUX.1 or FLUX.1 Kontext),such as "./FLUX.1-Kontext-dev"
    --learning_rate: learning rate, such as 1e-5
    --num_epochs : training epochs, such as 5
    --remove_prefix_in_ckpt : set to"pipe.dit." 
    --trainable_models : only train "dit" 
    --extra_inputs : extra keys except image and prompt, in our task it's "kontext_images" 
    --use_gradient_checkpointing : option to save memory, otherwise you have tons of GPU you can turn off
    --default_caption "Transform to normal map while maintaining original composition" \
    --batch_size : large batch size means faster convergence, but more memory need. It's recommend to set to "16" with 4 gpus (H200), if OOM, consider change to smaller values like 8, 4, 2, 1
    --output_path : output path of trained models, like "ckpts/kontext_normal/bs16_lora"
    --eval_file_list : in training we do evaluation to monitor the process, like "./data_split/nyu_normals/nyuv2_test2.txt" (depth task).
    --multi_res_noise : use multi resolution noise for faster convergence, refer to [Marigold](https://openaccess.thecvf.com/content/CVPR2024/papers/Ke_Repurposing_Diffusion-Based_Image_Generators_for_Monocular_Depth_Estimation_CVPR_2024_paper.pdf)
    --save_steps : num of steps to save model checkpoints, such as "200", after saving, it will do a complete evaluation on eval_file_list.
    --eval_steps : num of steps to do a evaluation on a subset(the first 10 files) of eval_file_list, such as "50" 
    --with_mask : when computing loss, just do in the mask area, usually in depth task it means the area with valid depth, and in normal task it means its , for more details please refer to `models/unified_dataset.py`
    --lora_base_model : the base model of lora, set to "dit"
    --lora_target_modules : set to "a_to_qkv,b_to_qkv,ff_a.0,ff_a.2,ff_b.0,ff_b.2,a_to_out,b_to_out,proj_out,norm.linear,norm1_a.linear,norm1_b.linear,to_qkv_mlp" 
    --lora_rank : the rank of lora, set to64 
    --align_to_opensource_format : change the keys of lora to the opensoure format, refer to `lora/flux_lora.py`
    --resume : resume from last checkpoint in the output_path (if not found, start from base model)
    --height, --width : training height and width of image
    --adamw8bit : use 8 bit AdamW optimizer to save memory
    --using_sqrt : use sqrt normalization for depth. (Other options: using_log / (left for blank means uniform normalization))
    --extra_loss: use extra loss to supervise training, such as "cycle_consistency_normal_estimation"
    --deterministic_flow : use deterministic rather than stochastic denoising process, refer to [FE2E](https://arxiv.org/abs/2509.04338) -->


# Training Arguments Explanation

This document provides a detailed explanation of the training arguments used in our scripts.

### Core Arguments
*   `--model_paths`: Path to the base model directory (e.g., `./FLUX.1-Kontext-dev`).
*   `--output_path`: Directory to save model checkpoints and logs.
*   `--learning_rate`: The learning rate for the optimizer (e.g., `1e-5`).
*   `--num_epochs`: Total number of training epochs.
*   `--batch_size`: Per-GPU batch size. Adjust based on your GPU memory.
*   `--resume`: Resume training from the latest checkpoint in `--output_path`.

### Dataset Arguments
*   `--dataset_base_path`: Comma-separated absolute paths to the training datasets. **The order matters.**
*   `--dataset_metadata_path`: Path to the metadata file (CSV or Parquet) containing relative file paths.
*   `--data_file_keys`: Column names in the metadata file to be used, e.g., `kontext_images,image`.
*   `--dataset_repeat`: Number of times to repeat a dataset within an epoch. Useful for balancing datasets of different sizes.
*   `--height`, `--width`: Target resolution for training images.

### Model & Training Strategy
*   `--trainable_models`: Specifies which parts of the model to train. For fine-tuning, set to `"dit"`.
*   `--extra_inputs`: Specifies additional input keys besides the main image and prompt. In our case, it's `"kontext_images"`.
*   `--default_caption`: The default text prompt used for training (e.g., `"Transform to normal map..."`).
*   `--multi_res_noise`: (Flag) Use multi-resolution noise for potentially faster convergence, inspired by Marigold.
*   `--with_mask`: (Flag) Compute the loss only on valid masked areas (e.g., where ground truth depth is available).
*   `--using_sqrt`: (Flag, Depth only) Use our theoretically optimal square-root normalization for depth.
*   `--extra_loss`: Name of the pixel-space consistency loss to apply (e.g., `"cycle_consistency_normal_estimation"`).
*   `--deterministic_flow`: (Flag) Use a fixed random seed for the initial noise to create a pseudo-deterministic path.

### LoRA Specific Arguments
*   `--lora_base_model`: The base model to which LoRA is applied, typically `"dit"`.
*   `--lora_target_modules`: Comma-separated list of modules to apply LoRA to.
*   `--lora_rank`: The rank of the LoRA decomposition matrices (e.g., `64`).
*   `--align_to_opensource_format`: (Flag) Save LoRA weights in a community-standard format.

### Performance & Logging
*   `--use_gradient_checkpointing`: (Flag) Enable to save GPU memory at the cost of a small slowdown.
*   `--adamw8bit`: (Flag) Use the 8-bit AdamW optimizer to reduce memory usage.
*   `--save_steps`: Save a full model checkpoint every N steps.
*   `--eval_steps`: Perform a quick evaluation on a small subset every N steps to monitor progress.
*   `--eval_file_list`: Path to the text file containing the list of images for evaluation during training.