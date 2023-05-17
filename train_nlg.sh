DIR=$(pwd)
DATA=${DIR}/data
MODEL=${DIR}/model

cd ../transformers/examples/pytorch/summarization

for model_name in google/t5-v1_1-xl # google/t5-v1_1-large google/t5-v1_1-base  # t5-small # t5-11b #t5-large # t5-small t5-base 
do
for mode in positive negative
do
TASK=anli_anlg_${mode}
python3 -m torch.distributed.launch --nproc_per_node=4 run_summarization.py \
    --model_name_or_path ${model_name} \
    --do_train \
    --do_eval \
    --do_predict \
    --train_file ${DATA}/${TASK}/train.json \
    --validation_file ${DATA}/${TASK}/dev.json \
    --test_file ${DATA}/${TASK}/test.json \
    --output_dir ${MODEL}/${TASK}_${model_name} \
    --overwrite_output_dir \
    --num_train_epochs 5 \
    --overwrite_cache \
    --per_device_train_batch_size=48 \
    --per_device_eval_batch_size=4 \
    --evaluation_strategy epoch \
    --predict_with_generate \
    --learning_rate 3e-5 \
    --save_strategy epoch \
    --warmup_steps 50 \
    --logging_steps 500 \
    --sharded_ddp simple \
    --load_best_model_at_end  \
    --lang 0
rm -rf ${MODEL}/${TASK}_${model_name}/checkpoint*
done
done
