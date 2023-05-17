import json
import argparse
from os import posix_fadvise


parser = argparse.ArgumentParser()
parser.add_argument('--positive_file', type=str)
parser.add_argument('--negative_file', type=str)
parser.add_argument('--output_file', type=str)
parser.add_argument('--input_file', type=str, default='data/anli/train_clean.jsonl')

args = parser.parse_args()

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
appear_set = set()
count = 0
for i in range(len(positive)):
    positive_list = eval(positive[i])
    for hypo_positive in positive_list:
        instance_dict = {
            'sentence1': ' '.join([src[i]['obs1'], classifier_sep_token, hypo_positive[0].strip()]),
            'sentence2': src[i]['obs2'],
            'label': good_label
        }
        count += 1   
        print(json.dumps(instance_dict), file=f_classifier)
    negative_list = eval(negative[i])
    for hypo_negative in negative_list:
        instance_dict = {
            'sentence1': ' '.join([src[i]['obs1'], classifier_sep_token, hypo_negative[0].strip()]),
            'sentence2': src[i]['obs2'],
            'label': bad_label
        }
        count += 1 
        print(json.dumps(instance_dict), file=f_classifier)
'''
for i in range(len(positive)):
    if positive[i] != '':
        instance_dict = {
            'sentence1': ' '.join([src[i]['obs1'], classifier_sep_token, positive[i].strip()]),
            'sentence2': src[i]['obs2'],
            'label': good_label
        }
        count += 1   
        print(json.dumps(instance_dict), file=f_classifier)
    if negative[i] != '':
        instance_dict = {
            'sentence1': ' '.join([src[i]['obs1'], classifier_sep_token, negative[i].strip()]),
            'sentence2': src[i]['obs2'],
            'label': bad_label
        }
        count += 1 
        print(json.dumps(instance_dict), file=f_classifier)
'''
print(count)