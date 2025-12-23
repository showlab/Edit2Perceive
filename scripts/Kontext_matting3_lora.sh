# export NCCL_P2P_LEVEL=2
# export NCCL_P2P_DISABLE=1
# export NCCL_IB_TIMEOUT=22
# export TORCH_NCCL_BLOCKING_WAIT=0

export TORCH_NCCL_TIMEOUT=1800000

accelerate launch --config_file configs/accelerate_config.yaml scripts/train.py \
    --dataset_base_path /mnt/nfs/workspace/syq/dataset/matting/composition-1k,/mnt/nfs/workspace/syq/dataset/matting/Distinctions-646,/mnt/nfs/workspace/syq/dataset/matting/AM-2k,/mnt/nfs/workspace/syq/dataset/matting/COCO-Matte \
    --dataset_metadata_path data_split/comp_matting/filenames_train.txt,data_split/Distinctions_matting/filenames_train.txt,data_split/AM_matting/filenames_train.txt,data_split/coco_matting/filenames_train.txt \
    --data_file_keys kontext_images,image \
    --model_paths ./FLUX.1-Kontext-dev \
    --learning_rate 1e-5 \
    --num_epochs 30 \
    --remove_prefix_in_ckpt pipe.dit. \
    --trainable_models dit \
    --extra_inputs kontext_images,trimap \
    --use_gradient_checkpointing \
    --multi_res_noise \
    --default_caption "Transform to matting map while maintaining original composition" \
    --with_mask \
    --batch_size 4 \
    --save_steps 50 \
    --matting_prompt points \
    --output_path ckpts/kontext_matting/bs16_cons_mixSDMatte_points_lora \
    --eval_file_list ./data_split/P3M_matting/filenames_val_NP.txt \
    --eval_steps 50 \
    --resume \
    --height 512 \
    --width 512 \
    --adamw8bit \
    --lora_base_model "dit" \
    --lora_target_modules "a_to_qkv,b_to_qkv,ff_a.0,ff_a.2,ff_b.0,ff_b.2,a_to_out,b_to_out,proj_out,norm.linear,norm1_a.linear,norm1_b.linear,to_qkv_mlp" \
    --lora_rank 64 \
    --align_to_opensource_format \
    --extra_loss cycle_consistency_matting_estimation
    # --extra_inputs kontext_images,trimap,visual_prompt_coords \
    # --extra_loss cycle_consistency_matting_estimation \

    # --with_mask  --use_coor_input \
    # --matting_prompt bbox \
    # --deterministic_flow


