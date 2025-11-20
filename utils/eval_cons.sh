#!/bin/bash
export CUDA_VISIBLE_DEVICES=4
# 定义日志文件名
LOG_FILE="eval_cons_diode.txt"

# 清空旧日志（可选，若想保留历史可删除此行）
> "$LOG_FILE"

# 定义要执行的所有命令（数组形式，每个元素是完整的 python 命令）
COMMANDS=(
  "python diffsynth/utils/eval_multiple_datasets.py --cur_step 2700 --state_dict models/train/kontext/flux/uni/step-2700.safetensors --datasets diode --model_root /mnt/nfs/share_model/FLUX.1-dev" 
  "python diffsynth/utils/eval_multiple_datasets.py --cur_step 1800 --state_dict models/train/kontext/flux/sqrt/step-1800.safetensors --datasets diode --model_root /mnt/nfs/share_model/FLUX.1-dev" 
  "python diffsynth/utils/eval_multiple_datasets.py --cur_step 3050 --state_dict models/train/kontext/flux/uni_cons/step-3050.safetensors --datasets diode --model_root /mnt/nfs/share_model/FLUX.1-dev" 
  "python diffsynth/utils/eval_multiple_datasets.py --cur_step 2200 --state_dict models/train/kontext/flux/sqrt/step-2200.safetensors --datasets diode --model_root /mnt/nfs/share_model/FLUX.1-dev" 
  "python diffsynth/utils/eval_multiple_datasets.py --cur_step 3800 --state_dict models/train/kontext/bs64_mask/step-3800.safetensors --datasets diode"
  "python diffsynth/utils/eval_multiple_datasets.py --cur_step 1850 --state_dict models/train/kontext/bs64_sqrt_mask/step-1850.safetensors --datasets diode"
  "python diffsynth/utils/eval_multiple_datasets.py --cur_step 4000 --state_dict models/train/kontext/bs64_mask/step-4000.safetensors --datasets diode"
#   "python diffsynth/utils/eval_multiple_datasets.py --cur_step 5300 --state_dict models/train/kontext/bs64_sqrt_cons/step-5300.safetensors --datasets diode"
)

# 循环执行每个命令
for cmd in "${COMMANDS[@]}"; do
  # 从命令中提取 state_dict 的值（正则匹配 --state_dict 后的路径）
  state_dict=$(echo "$cmd" | grep -oP '(?<=--state_dict\s)\S+')
  
  # 打印提示信息（可选，方便实时查看进度）
  echo "正在执行：$cmd"
  echo "对应的 state_dict：$state_dict"
  
  # 执行命令，捕获所有输出（stdout + stderr），并按格式追加到日志文件
  # 格式：[state_dict 值] 原始输出内容（每行都添加前缀）
  $cmd 2>&1 | while IFS= read -r line; do
    echo "[$state_dict] $line" >> "$LOG_FILE"
  done
  
  # 打印分隔符（可选，方便日志文件中区分不同命令的输出）
  echo "----------------------------------------" >> "$LOG_FILE"
done

echo "所有命令执行完毕，日志已保存到 $LOG_FILE"