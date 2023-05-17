import json
import argparse
import datetime
import time

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--output_file', type=str, default='')
parser.add_argument('--deberta_filter', action='store_true', default=False)
parser.add_argument('--fp16', action='store_true', default=False)
parser.add_argument('--deberta_model_path', type=str, default='microsoft/deberta-v2-xlarge-mnli')
parser.add_argument('--beam_search', action='store_true', default=False)
parser.add_argument('--deberta_filter_rerank', action='store_true', default=False)
parser.add_argument('--deberta_filter_max_prob', type=float, default=1.0)
parser.add_argument('--deberta_filter_min_prob', type=float, default=0.0)
parser.add_argument('--deberta_filter_dynamic_threshold', type=float, default=1.0)
parser.add_argument('--select_count', type=int, default=1)
parser.add_argument('--output_line', action='store_true', default=False)
parser.add_argument('--bert_nsp_filter', action='store_true', default=False)
parser.add_argument('--bert_nsp_rerank', action='store_true', default=False)
parser.add_argument('--rerank_by_comparing', action='store_true', default=False)
parser.add_argument('--constraints', action='store_true', default=False)
parser.add_argument('--negative', action='store_true', default=False)
parser.add_argument('--threshold_rate', type=float, default=0.999)
parser.add_argument('--topp', type=float, default=0.9)
parser.add_argument('--prob_threshold', type=float, default=0.9)
parser.add_argument('--infill_token', type=str, default='<extra_id_0>')
parser.add_argument('--src', type=str, default='anlg/test_compress.json')
parser.add_argument('--generated_candidates', type=str, default=None)
parser.add_argument('--model_type', type=str, default='t5')
parser.add_argument('--model_path', type=str, default='t5-large')

parser.add_argument('--tuned_bart', action='store_true', default=False)
parser.add_argument('--output_json', action='store_true', default=False)


args = parser.parse_args()

if args.generated_candidates is not None:
    with open(args.generated_candidates, 'r') as f:    
        raw_generated_candidates = f.readlines()
        generated_candidates = [eval(c) for c in raw_generated_candidates]
        
else:
    generated_candidates = None

output_file = '.'.join([
    'test',
    str(datetime.date.today()),
    args.model_type,
    args.model_path.split('/')[-1],
    'tuned_bart' if args.tuned_bart else '', \
    'deberta_filter' if args.deberta_filter else '', \
    'deberta_filter_rerank' if args.deberta_filter_rerank else '', \
    'bert_nsp_filter' if args.bert_nsp_filter else '', \
    'bert_nsp_rerank' if args.bert_nsp_rerank else '', \
    'rerank_by_comparing' if args.rerank_by_comparing else '', \
    'topp', str(args.topp)
]) if args.output_file == '' else args.output_file

output_full_res_name = 'full_res.' + output_file

f_output = open('res/'+output_file + '.txt', 'w')
last_len = 0
f_full_res = open('res/'+output_full_res_name + '.txt', 'w')



contradiction_label = 0
entailment_label = 2

with open(args.src, 'r') as f:
    if not args.output_json:
        test_text = f.readlines()
        test_text = [json.loads(text) for text in test_text]
    else:
        test = json.load(f)
        test_text = []
        for key in test:
            instance = test[key]
            instance['story_id'] = key
            test_text.append(instance)
        

test_num = len(test_text)

from text_infiller import *
infiller = TextInfiller(model_type=args.model_type, model_path=args.model_path, deberta_model_path=args.deberta_model_path, \
    deberta_filter=args.deberta_filter, deberta_bad_label=1 if args.negative else 0, bert_nsp_filter=args.bert_nsp_filter, \
    rerank_by_comparing=args.rerank_by_comparing, fp16=args.fp16, infill_token=args.infill_token)
output_json = dict()

i, j = 0, 0 
instance_count = len(test_text)
full_lists = []
output_count = 0
while i < instance_count:
    test_instance = test_text[i]
    while j+1 < instance_count and test_instance['obs1'] == test_text[j+1]['obs1'] and test_instance['obs2'] == test_text[j+1]['obs2']: 
        j += 1
    same_input_count = j - i + 1 if args.select_count == -1 else args.select_count
    i = j + 1


    story_id = test_instance['story_id']
    input_left, input_right = test_instance['obs1'].strip().capitalize(), test_instance['obs2'].strip().capitalize()
    if input_left[-1].isalpha():
        input_left += '.'
    if input_right[-1].isalpha():
        input_right += '.'
    constraint = ''
    if args.beam_search:
        res, full_list = infiller.beam_search(input_left=input_left, input_right=input_right)
        print(res[0], file=f_output)    
        print('no.', i, input_left, '*', res[0], '*', input_right, res[2])
        if args.generated_candidates is None:
            f_full_res.write(str(full_list))
            f_full_res.write('\n')
        continue
    else:
        if args.generated_candidates is None or generated_candidates == []:
            res, full_list = infiller.predict_list(least_num=args.select_count, max_iter=2, input_left=input_left, input_right=input_right, \
                batch_size=args.batch_size, topp=args.topp, deberta_filter=args.deberta_filter, deberta_rerank=args.deberta_filter_rerank, \
                bert_nsp_filter=args.bert_nsp_filter, bert_nsp_rerank=args.bert_nsp_rerank, prob_threshold=args.prob_threshold)
        else:
            full_list = generated_candidates.pop(0)
            if args.deberta_filter:
                full_list_filter = infiller.run_deberta_filter(input_left=input_left, input_right=input_right, outputs_valid=full_list, dynamic_threshold=args.deberta_filter_dynamic_threshold)
                if full_list_filter == []:
                    full_list_filter = infiller.run_deberta_filter(input_left=input_left, input_right=input_right, outputs_valid=full_list, prob_threshold=0.5)
                    if full_list_filter == []:
                        full_list_filter = full_list
                full_list = full_list_filter
                if args.deberta_filter_max_prob < 1.0:
                    full_list = list(filter(lambda item: item[2] < args.deberta_filter_max_prob, full_list))
                res = max(full_list, key=lambda item: item[1]) if args.deberta_filter_rerank is False else max(full_list, key=lambda item: item[-1])
            else:
                res = full_list[0]
    print('no.', i, input_left, '*', res[0], '*', input_right, res[1:], sep='\t')
    if args.select_count > 1:
        print(full_list[:args.select_count], file=f_output)
    else:
        print(res[0], file=f_output)
        output_json[story_id] = res[0]
   
    if args.generated_candidates is None:
        f_full_res.write(str(full_list))
        f_full_res.write('\n')

if args.output_json:
    with open(output_file + '.json', 'w') as fp:
        output_json = json.dumps(output_json)
        print(output_json, file=fp)