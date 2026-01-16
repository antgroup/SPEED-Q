export CODE_DIR="$(pwd)"

# ############################### internvl3-1b ###############################
export BASE_PATH="/home/usr/InternVL-training/speedq_dir/logs/internvl3-1b"

# # 2bit 16 16
EVAL_MODEL_PATH="${BASE_PATH}/int2-bilevel-g16-g16-sapmle10-datasetv3-full-8e-6-multi/checkpoint-10000"
SAVE_MERGED_PATH="${BASE_PATH}/int2-bilevel-g16-g16-sapmle10-datasetv3-full-8e-6-multi/fake_quant/checkpoint-10000"
export Q_BIT="2"
export Q_GROUP_SIZE="16"
export QQ_GROUP_SIZE="16"
export QUANT_VIT="false"
export QUANT_LLM="true"

# 4bit 32 128
EVAL_MODEL_PATH="${BASE_PATH}/int4-bilevel-g32-g128-sapmle10-datasetv3-full-8e-6-multi/checkpoint-10000"
SAVE_MERGED_PATH="${BASE_PATH}/int4-bilevel-g32-g128-sapmle10-datasetv3-full-8e-6-multi/fake_quant/checkpoint-10000"
export Q_BIT="4"
export Q_GROUP_SIZE="32"
export QQ_GROUP_SIZE="128"
export QUANT_VIT="false"
export QUANT_LLM="true"

python save_fake_quant.py \
    $EVAL_MODEL_PATH \
    --wbits $Q_BIT \
    --groupsize $Q_GROUP_SIZE \
    --qq_groupsize $QQ_GROUP_SIZE \
    --quant_vit "${QUANT_VIT}" \
    --quant_llm "${QUANT_LLM}" \
    --save_merged $SAVE_MERGED_PATH/quant \
    --save_quant_info $SAVE_MERGED_PATH/quant_details

# copy some base model config to the folder
cp -r /home/usr/templete_internvl3_1b/* $SAVE_MERGED_PATH/quant/