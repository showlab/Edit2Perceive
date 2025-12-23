accelerate launch --config_file configs/accelerate_config.yaml scripts/train.py \
    --dataset_base_path "/mnt/nfs/workspace/syq/dataset/Hypersim/processed_depth,/mnt/nfs/workspace/syq/dataset/vkitti2" \
    --dataset_metadata_path "./data_split/hypersim_depth/filename_list_train_filtered2.txt,./data_split/vkitti_depth/vkitti_train.txt" \
    --data_file_keys "kontext_images,image" \
    --model_paths "./FLUX.1-Kontext-dev" \
    --learning_rate "3e-6" \
    --num_epochs "8" \
    --remove_prefix_in_ckpt "pipe.dit." \
    --trainable_models "dit" \
    --extra_inputs "kontext_images" \
    --use_gradient_checkpointing \
    --default_caption "Transform to depth map while maintaining original composition" \
    --batch_size 2 \
    --output_path "ckpts/kontext_depth/bs64_sqrt_lora" \
    --eval_file_list "./data_split/nyu_depth/labeled/filename_list_test.txt" \
    --multi_res_noise \
    --save_steps 200 \
    --eval_steps 10 \
    --with_mask \
    --depth_normalization sqrt \
    --using_sqrt \
    --lora_base_model "dit" \
    --lora_target_modules "a_to_qkv,b_to_qkv,ff_a.0,ff_a.2,ff_b.0,ff_b.2,a_to_out,b_to_out,proj_out,norm.linear,norm1_a.linear,norm1_b.linear,to_qkv_mlp" \
    --lora_rank 64 \
    --align_to_opensource_format \
    --height 512 \
    --width 768 \
    --resume
    # --deterministic_flow \
    # --extra_loss "cycle_consistency_depth_estimation" 

    # --extra_loss_start_epoch 0 \
    # --using_sqrt \
    # --resume \


