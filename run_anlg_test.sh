
GENERATOR_PATH=model/anli_anlg_positive_google/t5-v1_1-xl
TEST_FILE=data/anlg/test_cleanup.jsonl
CLASSIFIER_PATH=model/anli_classifier_aug_xl_high_prob_2x/
python3 anlg.py --src ${TEST_FILE}  --output_file anlg_test_t5_xl_aug_filter --model_path ${GENERATOR_PATH} --deberta_model_path ${CLASSIFIER_PATH} --deberta_filter & 


# submission
GENERATOR_PATH=model/anli_anlg_positive_google/t5-v1_1-xl
TEST_FILE=data/anlg/test_cleanup_no_label.json
CLASSIFIER_PATH=model/anli_classifier_aug_xl_high_prob_2x/checkpoint-5500
python3 anlg.py --src ${TEST_FILE} --output_file anlg_test_t5_xl_submission --model_path ${GENERATOR_PATH} --deberta_model_path ${CLASSIFIER_PATH} --deberta_filter --output_json &

