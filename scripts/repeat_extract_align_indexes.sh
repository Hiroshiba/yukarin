#!/usr/bin/env bash

NAME=${1}
GPU=${2}
LOOP=${3}
OUTPUT=${4}
CONFIG=${5}

mkdir ${OUTPUT}

CONFIG_JSON=`cat ${CONFIG}`
CONFIG_JSON=`echo ${CONFIG_JSON} | jq ".project.name = \"${NAME}\""`
CONFIG_JSON=`echo ${CONFIG_JSON} | jq ".train.gpu = ${GPU}"`

INPUT_GLOB=`echo ${CONFIG_JSON} | jq -r '.dataset.input_glob'`
TARGET_GLOB=`echo ${CONFIG_JSON} | jq -r '.dataset.target_glob'`

# alignment
OUTPUT_ALIGNMENT=${OUTPUT}/${NAME}_align_indexes_0/
PYTHONPATH=`pwd` python scripts/extract_align_indexes.py \
    -i1 "${INPUT_GLOB}" \
    -i2 "${TARGET_GLOB}" \
    -o ${OUTPUT_ALIGNMENT} \

for i_loop in $(seq 1 ${LOOP}); do
    # train
    OUTPUT_TRAIN=${OUTPUT}/${NAME}_train_${i_loop}/
    mkdir ${OUTPUT_TRAIN}

    CONFIG_JSON=`echo ${CONFIG_JSON} | jq ".dataset.indexes_glob = \"${OUTPUT_ALIGNMENT}/*.npy\""`
    CONFIG=${OUTPUT_TRAIN}/config.json
    echo ${CONFIG_JSON} > ${CONFIG}

    PYTHONPATH=`pwd` python train.py ${CONFIG} ${OUTPUT_TRAIN}

    MODEL=`find ${OUTPUT_TRAIN} -name 'predictor_*.npz' | sort -n | tail -n1`
    CONFIG_JSON=`echo ${CONFIG_JSON} | jq ".train.pretrained_model = \"${MODEL}\""`  # for next loop

    # convert
    OUTPUT_CONVERTED=${OUTPUT}/${NAME}_converted_${i_loop}/

    CONFIG=${OUTPUT_TRAIN}/config.json
    PYTHONPATH=`pwd` python scripts/convert_acoustic_feature.py \
        -i "${INPUT_GLOB}" \
        -o ${OUTPUT_CONVERTED} \
        -vcm ${MODEL} \
        -vcc ${CONFIG} \

    # alignment
    OUTPUT_ALIGNMENT=${OUTPUT}/${NAME}_align_indexes_${i_loop}/
    PYTHONPATH=`pwd` python scripts/extract_align_indexes.py \
        -i1 "${OUTPUT_CONVERTED}/*.npy" \
        -i2 "${TARGET_GLOB}" \
        -o ${OUTPUT_ALIGNMENT} \

    # remove
    rm -r ${OUTPUT_CONVERTED}
done
