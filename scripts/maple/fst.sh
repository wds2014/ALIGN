#!/bin/bash

cd ../..

# custom config
DATA="/data4/wds/dataset/CoOpData/"
TRAINER=MaPLe

CFG=vit_b16_c2_ep5_batch4_2ctx
#DATASET=$1

#for DATASET in caltech101 dtd eurosat fgvc_aircraft
#for DATASET in food101 oxford_flowers oxford_pets stanford_cars
for DATASET in ucf101
do
for SHOTS in 1 2 4 8 16
do
for SEED in 1 2 3
do
DIR=output/fewshot/${DATASET}/${TRAINER}_2/shots_${SHOTS}/seed${SEED}
if [ -d "$DIR" ]; then
    echo "Results are available in ${DIR}. Resuming..."
    python train.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
    --output-dir ${DIR} \
    DATASET.NUM_SHOTS ${SHOTS}
else
    echo "Run this job and save the output to ${DIR}"
    python train.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
    --output-dir ${DIR} \
    DATASET.NUM_SHOTS ${SHOTS}
fi
done
done
done