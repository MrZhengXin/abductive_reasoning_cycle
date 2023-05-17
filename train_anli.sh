DIR=$(pwd)
DATA=${DIR}/data
MODEL=${DIR}/model
TASK=anli_classifier
CLASSIFIER=microsoft/deberta-v3-large
SRC=${DIR}/data/anli/train_clean.jsonl
export MASTER_PORT=9933

CLASSIFIER_PATH=${MODEL}/${TASK}
cd ../transformers/examples/pytorch/text-classification
python3 -m torch.distributed.launch --nproc_per_node=4 run_glue.py \
    --model_name_or_path ${CLASSIFIER} \
    --train_file ${DATA}/${TASK}/train_merge.json \
    --validation_file ${DATA}/${TASK}/dev.json \
    --test_file ${DATA}/${TASK}/test.json \
    --do_train \
    --do_eval \
    --do_predict \
    --no_pad_to_max_length \
    --per_device_train_batch_size 24 \
    --learning_rate 6e-6 \
    --num_train_epochs 5 \
    --warmup_ratio 0.06 \
    --output_dir ${CLASSIFIER_PATH} \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy steps \
    --eval_steps 500 \
    --save_strategy steps \
    --save_steps 500 \
    --logging_steps 500 \
    --overwrite_cache
    