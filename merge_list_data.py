import json
import argparse
import random


parser = argparse.ArgumentParser()
parser.add_argument('--positive_file', type=str)
parser.add_argument('--negative_file', type=str)
parser.add_argument('--output_file', type=str)
parser.add_argument('--positive_only', action='store_true', default=False)
parser.add_argument('--prob_threshold', type=float, default=0.99)
parser.add_argument('--augment_ratio', type=int, default=1)
parser.add_argument('--augment_count', type=int, default=-1)
parser.add_argument('--input_file', type=str, default='data/anli/train_clean.jsonl')

args = parser.parse_args()

random.seed()

good_label, bad_label = 1, 0
classifier_sep_token = '[SEP]'
generator_sep_token = '<extra_id_0>'

with open(args.input_file, 'r') as f:
    src = f.readlines()
    src = [json.loads(s) for s in src]

with open(args.positive_file, 'r') as f:
    positive = f.readlines()

with open(args.negative_file, 'r') as f:
    negative = f.readlines()


src_aug = []
f_classifier = open(args.output_file, 'a')
i, j, k = 0, 0, 0 # 7371, 7371
instance_count, list_count = len(src), min(len(positive), len(negative))
for k in range(list_count):
    if i >= instance_count:
        print(i, j, k)
        break
    while j+1 < instance_count and src[i]['obs1'] == src[j+1]['obs1'] and src[i]['obs2'] == src[j+1]['obs2']: 
        j += 1
    positive_list = eval(positive[k])
    # positive_list = sorted(positive_list, key=lambda item: -item[-1])
    positive_list = list(filter(lambda x: x[-1] > args.prob_threshold, positive_list))
    positive_list = sorted(positive_list, key=lambda item: -item[1])
    cnt = (j - i + 1) * args.augment_ratio if args.augment_count == -1 else args.augment_count
    # print('positive:')
    # print(*positive_list[:cnt], sep='\n')
    for p in positive_list[:cnt]:
        hyp = p[0].strip()
        if hyp == '':
            continue
        instance_dict = {
            'sentence1': ' '.join([src[i]['obs1'], classifier_sep_token, hyp]),
            'sentence2': src[i]['obs2'],
            'label': good_label,
        }
        print(json.dumps(instance_dict), file=f_classifier)
    if not args.positive_only:
        negative_list =  eval(negative[k])
        # negative_list = sorted(negative_list, key=lambda item: -item[-1])
        negative_list = list(filter(lambda x: x[-1] > args.prob_threshold, negative_list))
        negative_list = sorted(negative_list, key=lambda item: -item[1])
        # print('\n\nnegative:')
        # print(*negative_list[:cnt], sep='\n')
        # input()

        for n in negative_list[:cnt]:
            hyp = n[0].strip()
            if hyp == '':
                continue           
            instance_dict = {
                'sentence1': ' '.join([src[i]['obs1'], classifier_sep_token, hyp]),
                'sentence2': src[i]['obs2'],
                'label': bad_label,
            }   
            print(json.dumps(instance_dict), file=f_classifier)
    
    i = j + 1
f_classifier.close()
with open(args.output_file, 'r') as f:
    data_list = f.readlines()
    print(len(data_list))

random.shuffle(data_list)
f_classifier = open(args.output_file, 'w')
print(*data_list, sep='',file=f_classifier)
