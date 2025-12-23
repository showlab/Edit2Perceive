# export NCCL_P2P_LEVEL=2
# export NCCL_P2P_DISABLE=1
# export NCCL_IB_TIMEOUT=22
# export TORCH_NCCL_BLOCKING_WAIT=0

export TORCH_NCCL_TIMEOUT=1800

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
    --batch_size 16 \
    --save_steps 200 \
    --matting_prompt points \
    --output_path ckpts/kontext_matting/bs16_cons_mixSDMatte_points \
    --eval_file_list ./data_split/P3M_matting/filenames_val_NP.txt \
    --eval_steps 50 \
    --resume \
    --adamw8bit \
    --dataset_num_workers 16 \
    --resume \
    --extra_loss cycle_consistency_cycle_consistency_matting_estimation \
    --height 512 \
    --width 512
    # --extra_inputs kontext_images,trimap,visual_prompt_coords \
    # --extra_loss cycle_consistency_matting_estimation \

    # --with_mask  --use_coor_input \
    # --matting_prompt bbox \
    # --deterministic_flow


