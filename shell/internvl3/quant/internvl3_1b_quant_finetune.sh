set -x

GPUS=${GPUS:-8}
BATCH_SIZE=${BATCH_SIZE:-128}
PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE:-4}
GRADIENT_ACC=$((BATCH_SIZE / PER_DEVICE_BATCH_SIZE / GPUS))


export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export MASTER_PORT=34302
export TF_CPP_MIN_LOG_LEVEL=3
export LAUNCHER=pytorch

bits=${1}
q_group_size=${2}
qq_group_size=${3}
quant_level=${4}
calkd_steps=${5}
model_path=${6}
teacher_model_path=${7}
dataset=${8}
output_dir=${9}
clip=${10}
quant_vit=${11}
quant_mlp=${12}
quant_llm=${13}
freeze_vit=${14}
freeze_mlp=${15}
freeze_llm=${16}
sample_num=${17}
use_distill=${18}
mix_quant=${19}
max_seq_length=${20}
quant_vit_path=${21}


if [ ! -d "$output_dir" ]; then
  mkdir -p "$output_dir"
fi

# number of gpus: 8
# batch size per gpu: 4
# gradient accumulation steps: 4
# total batch size: 128
# epoch: 1
torchrun \
  --nnodes=1 \
  --node_rank=0 \
  --master_addr=127.0.0.1 \
  --nproc_per_node=${GPUS} \
  --master_port=${MASTER_PORT} \
  internvl/train/internvl_chat_quant_finetune.py \
  --clip ${clip} \
  --quant_vit_path ${quant_vit_path} \
  --quant_type int${bits}-${quant_level} \
  --mix_quant ${mix_quant} \
  --q_group_size ${q_group_size} \
  --qq_group_size ${qq_group_size} \
  --calkd_steps ${calkd_steps} \
  --model_name_or_path ${model_path} \
  --teacher_model_path ${teacher_model_path} \
  --output_dir ${output_dir} \
  --meta_path ${dataset} \
  --quant_vit "${quant_vit}" \
  --quant_mlp "${quant_mlp}" \
  --quant_llm "${quant_llm}" \
  --freeze_llm "${freeze_llm}" \
  --freeze_mlp "${freeze_mlp}" \
  --use_distill "${use_distill}" \
  --freeze_backbone "${freeze_vit}" \
  --per_device_train_batch_size ${PER_DEVICE_BATCH_SIZE} \
  --gradient_accumulation_steps ${GRADIENT_ACC} \
  --conv_style "internvl2_5" \
  --use_fast_tokenizer False \
  --overwrite_output_dir False \
  --sample_each_dataset ${sample_num} \
  --force_image_size 448 \
  --max_dynamic_patch 6 \
  --down_sample_ratio 0.5 \
  --drop_path_rate 0.1 \
  --vision_select_layer -1 \
  --dataloader_num_workers 4 \
  --bf16 True \
  --num_train_epochs 1 \
  --evaluation_strategy "no" \
  --save_strategy "steps" \
  --save_steps 5000 \
  --save_total_limit 1 \
  --learning_rate 8e-6 \
  --weight_decay 0.01 \
  --warmup_ratio 0.03 \
  --lr_scheduler_type "cosine" \
  --logging_steps 1 \
  --max_seq_length ${max_seq_length} \
  --do_train True \
  --grad_checkpoint True \
  --group_by_length True \
  --dynamic_image_size True \
  --use_thumbnail True \
  --ps_version 'v2' \
  --deepspeed "config/zero_stage1_config.json" \
  --report_to "tensorboard" \
  2>&1 | tee -a "${output_dir}/training_log.txt"
