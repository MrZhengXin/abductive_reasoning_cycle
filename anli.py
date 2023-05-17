from transformers import AutoTokenizer, AutoModelForSequenceClassification
import json
import torch
import argparse




parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str)
parser.add_argument('--second_model', type=str, default='')
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--predict', action='store_true')
parser.add_argument('--source', type=str, default='data/anli/test.jsonl')
args = parser.parse_args()

tokenizer = AutoTokenizer.from_pretrained(args.model)
model = AutoModelForSequenceClassification.from_pretrained(args.model).cuda()
second_model = AutoModelForSequenceClassification.from_pretrained(args.second_model).cuda() if args.second_model != '' else None
deberta_bad_label = 0

with open(args.source, 'r') as f:
    test_text = f.readlines()
    test_text = [json.loads(text) for text in test_text]


test_num =len(test_text)

batch_size = args.batch_size
choices = []
softmax = torch.nn.Softmax(dim=1)
for i in range(0, test_num, batch_size):
    len_batch = batch_size if i + batch_size <= test_num else test_num - i
    with torch.no_grad():
        pt_batch = tokenizer(
            [ [text['obs1'] + ' [SEP] ' + text['hyp1'].capitalize(), text['obs2'] ] for text in test_text[i: i+batch_size]] +
            [ [text['obs1'] + ' [SEP] ' + text['hyp2'].capitalize(), text['obs2'] ] for text in test_text[i: i+batch_size]],
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt"
        )
        for k in pt_batch.keys():
            pt_batch[k] = pt_batch[k].cuda()
        logits = model(**pt_batch).logits
        if args.second_model != '':
            second_logits = second_model(**pt_batch).logits
        probs = softmax(logits)
        if args.second_model != '':
            second_logits = second_model(**pt_batch).logits
            probs += softmax(second_logits)
        prob_hyp1 = probs[:len_batch]
        prob_hyp2 = probs[len_batch:]
        prob = prob_hyp2 - prob_hyp1 

        nli_labels = prob.argmax(dim=1)
        choices.append(nli_labels)

choices = torch.hstack(choices)

if args.predict:
    with open('res/anli_prediction.txt', 'w') as f:
        print(*(choices+1).tolist(), sep='\n', file=f)
else:
    with open('data/anli/test-labels.lst', 'r') as f:
        test_label = f.readlines()
        test_label = [int(label) - 1 for label in test_label]

    acc = (choices == torch.tensor(test_label).cuda()).sum() / test_num
    print(args.model, args.second_model, acc.data.cpu().item(), sep='\t')