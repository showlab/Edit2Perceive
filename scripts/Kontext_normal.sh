export TORCH_NCCL_TIMEOUT=1800
accelerate launch --config_file configs/accelerate_config.yaml scripts/train.py \
    --dataset_base_path "/mnt/nfs/workspace/syq/dataset/Hypersim/processed_normal,/mnt/nfs/workspace/syq/dataset/InteriorVerse/processed_normal,/mnt/nfs/workspace/syq/dataset/sintel" \
    --dataset_metadata_path "./data_split/hypersim_normals/hypersim_filtered_all_checked.txt,./data_split/interiorverse_normals/interiorverse_filtered_all.txt,./data_split/sintel_normals/sintel_filtered.txt" \
    --data_file_keys "kontext_images,image" \
    --model_paths "./FLUX.1-Kontext-dev" \
    --learning_rate "1e-4" \
    --num_epochs "8" \
    --remove_prefix_in_ckpt "pipe.dit." \
    --trainable_models "dit" \
    --extra_inputs "kontext_images" \
    --use_gradient_checkpointing \
    --default_caption "Transform to normal map while maintaining original composition" \
    --batch_size "16" \
    --output_path "ckpts/kontext_normal/bs16_deter" \
    --eval_file_list "./data_split/nyu_normals/nyuv2_test2.txt" \
    --multi_res_noise \
    --save_steps "200" \
    --eval_steps "50" \
    --with_mask \
    --resume \
    --height 512 \
    --width 768

    # --extra_loss "cycle_consistency_normal_estimation" \
    # --deterministic_flow
    # --using_log
    # --using_log \
    # --adamw8bit \


