


export SRC=data/anli/train_clean.jsonl
export DEBERTA_MODEL_PATH=model/anli_classifier
export MODEL_PATH=model/anli_anlg_positive_google/t5-v1_1-xl
CUDA_VISIBLE_DEVICES=3 python3 anlg.py --src ${SRC} --model_path ${MODEL_PATH} --deberta_model_path ${DEBERTA_MODEL_PATH} --deberta_filter --output_file anlg_positive_t5_xl_99 --prob_threshold 0.99 &

export MODEL_PATH=model/anli_anlg_negative_google/t5-v1_1-xl
CUDA_VISIBLE_DEVICES=2 python3 anlg.py --src ${SRC} --model_path ${MODEL_PATH} --deberta_model_path ${DEBERTA_MODEL_PATH} --deberta_filter --output_file anlg_negative_t5_xl_99  --prob_threshold 0.99 --negative &


export ratio=2
DIR=$(pwd)
DATA=${DIR}/data
TASK=anli_classifier
i=0

cp ${DATA}/${TASK}/train_round_${i}.json ${DATA}/${TASK}/train_xl_round_$((i+1))_aug_99_${ratio}x.json
python3 merge_list_data.py --positive_file res/full_res.anlg_positive_t5_xl_99.txt --negative_file res/full_res.anlg_negative_t5_xl_99.txt --output_file ${DATA}/${TASK}/train_xl_round_$((i+1))_aug_99_${ratio}x.json --augment_ratio ${ratio} --input_file ${SRC}

cd ../transformers/examples/pytorch/text-classification

DATA=${DIR}/data
MODEL=${DIR}/model
TASK=anli_classifier
CLASSIFIER=microsoft/deberta-v3-large
export MASTER_PORT=9933
CLASSIFIER_PATH=${MODEL}/${TASK}_aug_xl_99_${ratio}x
python3 -m torch.distributed.launch --nproc_per_node=4 run_glue.py \
    --model_name_or_path ${CLASSIFIER} \
    --train_file ${DATA}/${TASK}/train_xl_round_$((i+1))_aug_99_${ratio}x.json \
    --validation_file ${DATA}/${TASK}/dev.json \
    --do_train \
    --do_eval \
    --max_seq_length 256 \
    --no_pad_to_max_length \
    --per_device_train_batch_size 128 \
    --learning_rate 6e-6 \
    --num_train_epochs 10 \
    --warmup_ratio 0.06 \
    --output_dir ${CLASSIFIER_PATH} \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy steps \
    --eval_steps 500 \
    --save_strategy steps \
    --save_steps 500 \
    --logging_steps 500 \
    --overwrite_cache \
    --overwrite_output_dir \
    --load_best_model_at_end

cd ${DIR}
for checkpoint in ${CLASSIFIER_PATH}/checkpoint*
do
python3 anli.py --model ${checkpoint}  >> res/anli_test_res.txt
done


