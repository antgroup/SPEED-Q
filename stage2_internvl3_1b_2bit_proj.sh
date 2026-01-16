export CODE_DIR="$(pwd)"
export SAVE_DIR="/home/usr/InternVL-training/speedq_dir"

export MODEL_NAME="internvl3-1b"
export MODEL_PATH="/home/usr/VLM-Model/OpenGVLab__InternVL3-1B"
export TEACHER_MODEL_NAME="internvl3-1b"
export TEACHER_MODEL_PATH="/home/usr/VLM-Model/OpenGVLab__InternVL3-1B"

export QUANT_VIT_PATH="/home/usr/VLM-Model/internvl3_1b_2bit_vit_qdq.pth"

# 2bit
export Q_BIT="2"
export Q_GROUP_SIZE="16"
export QQ_GROUP_SIZE="16"
export QUANT_LEVEL="bilevel"
export CALKD_STEPS="100"
export DATASET="data/training_datasets.json"
export SAMPLE_NUM="1" # percent
export TASK_NAME_SUFFIX="full-8e-6-projector"

# settings
export QUANT_VIT="false"
export QUANT_MLP="false"
export QUANT_LLM="false"

export FREEZE_VIT="true"
export FREEZE_MLP="false"
export FREEZE_LLM="true"

export USE_DISTILL="false"
export MIX_QUANT="false"
export MAX_SEQ_LENGTH="8192"

JSON_NAME=$(basename "${DATASET}")
DATA_NAME="${JSON_NAME%.json}"

echo "The data json name is: $DATA_NAME"

# gpu settings
export GPU_INDEX="0"
export GPUS="8"
export BATCH_SIZE="64"
export PER_DEVICE_BATCH_SIZE="1"

# Train
echo "#################### Training the proj. ####################"

GPUS=$GPUS BATCH_SIZE=$BATCH_SIZE PER_DEVICE_BATCH_SIZE=$PER_DEVICE_BATCH_SIZE sh shell/internvl3/quant/internvl3_1b_quant_finetune.sh \
    $Q_BIT \
    $Q_GROUP_SIZE \
    $QQ_GROUP_SIZE \
    $QUANT_LEVEL \
    $CALKD_STEPS \
    $MODEL_PATH \
    $TEACHER_MODEL_PATH \
    $DATASET \
    ${SAVE_DIR}/logs/${MODEL_NAME}/int${Q_BIT}-${QUANT_LEVEL}-g${Q_GROUP_SIZE}-g${QQ_GROUP_SIZE}-sapmle${SAMPLE_NUM}-${DATA_NAME}-${TASK_NAME_SUFFIX}/ \
    "None" \
    ${QUANT_VIT} \
    ${QUANT_MLP} \
    ${QUANT_LLM} \
    ${FREEZE_VIT} \
    ${FREEZE_MLP} \
    ${FREEZE_LLM} \
    ${SAMPLE_NUM} \
    ${USE_DISTILL} \
    ${MIX_QUANT} \
    ${MAX_SEQ_LENGTH} \
    ${QUANT_VIT_PATH}

echo "#################### Training the proj. complete ####################"