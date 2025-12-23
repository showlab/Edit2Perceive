accelerate launch --config_file configs/accelerate_config.yaml scripts/train.py \
    --dataset_base_path "/mnt/nfs/workspace/syq/dataset/Hypersim/processed_depth,/mnt/nfs/workspace/syq/dataset/vkitti2" \
    --dataset_metadata_path "./data_split/hypersim_depth/filename_list_train_filtered2.txt,./data_split/vkitti_depth/vkitti_train.txt" \
    --data_file_keys "kontext_images,image" \
    --model_paths "./FLUX.1-Kontext-dev" \
    --learning_rate "1e-5" \
    --num_epochs "8" \
    --remove_prefix_in_ckpt "pipe.dit." \
    --trainable_models "dit" \
    --extra_inputs "kontext_images" \
    --use_gradient_checkpointing \
    --default_caption "Transform to depth map while maintaining original composition" \
    --batch_size "4" \
    --output_path "ckpts/kontext/bs64_sqrt_cons" \
    --eval_file_list "./data_split/nyu_depth/labeled/filename_list_test.txt" \
    --multi_res_noise \
    --save_steps "200" \
    --eval_steps "50" \
    --with_mask \
    --depth_normalization sqrt \
    --dataset_num_workers "16" \
    --extra_loss "cycle_consistency_depth_estimation" \
    --adamw8bit \
    --using_sqrt
    # --deterministic_flow
    # --extra_loss_start_epoch 0 \
    # --using_sqrt \
    # --resume \


