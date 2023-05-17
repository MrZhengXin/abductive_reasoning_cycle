from transformers import BertTokenizer, BertForNextSentencePrediction
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import BartForConditionalGeneration, BartTokenizer
from transformers import T5Tokenizer, T5ForConditionalGeneration
import gc
import re
import time

import torch
import sacrebleu
from nltk.tokenize import sent_tokenize



deberta_model_name = "microsoft/deberta-v2-xlarge-mnli"
compare_checkpoint = ''

class TextInfiller:
    def __init__(self, model_type='t5', model_path='t5-large', deberta_filter=False, deberta_model_path=deberta_model_name, deberta_bad_label=0, bert_nsp_filter=False, rerank_by_comparing=False, compare_model_path=compare_checkpoint, fp16=False, infill_token='<extra_id_0>', *args, **kwargs):
        #self.model = torch.hub.load('pytorch/fairseq', 'bart.large')
        self.model_type = model_type
        if model_type == 'bart':
            self.model = torch.hub.load('pytorch/fairseq', 'bart.large').eval().cuda()
            if fp16:
                self.model = self.model.half()            
            self.tokenizer = BartTokenizer.from_pretrained("facebook/bart-large")
            self.infill_token = '<mask>'
            #self.model = BARTModel.from_pretrained('../fairseq/bart.large', checkpoint_file='model.pt')
        #self.model = BARTModel.from_pretrained('../fairseq/checkpoints', checkpoint_file='checkpoint6.pt')
        if model_type == 't5':
            self.model = T5ForConditionalGeneration.from_pretrained(model_path).eval().cuda()
            if fp16:
                self.model = self.model.half()
            self.tokenizer = T5Tokenizer.from_pretrained(model_path)
            self.extra_id_1_token = 32098
            self.infill_token = infill_token
            # print('load', model_type, model_path)

        if bert_nsp_filter:
            self.bert_tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
            self.bert = BertForNextSentencePrediction.from_pretrained('bert-large-uncased')
            self.bert.cuda()
        if deberta_filter:
            self.deberta_model = AutoModelForSequenceClassification.from_pretrained(deberta_model_path).eval().cuda() #.half()
            self.deberta_bad_label = deberta_bad_label
            self.deberta_tokenizer = AutoTokenizer.from_pretrained(deberta_model_path)
            # print('load deberta model', deberta_model_path)
        if rerank_by_comparing:
            self.compare_model = AutoModelForSequenceClassification.from_pretrained(compare_model_path).cuda()
            # print('load compare model', compare_model_path)
        self.contradiction_label = 0
        self.entailment_label = 2
        self.good_label = 1
        self.bad_label = 0
        self.softmax = torch.nn.Softmax(dim=1)

    def valid_hypothesis(self, output, input_left, input_right, full_sentence=True, bleu_threshold=65):
        h, logp = output
        if h == '':
            return ''
        if self.model_type == 'bart':
            h = h.replace(input_left, '').replace(input_right, '')
        h = re.sub(r'\.+', '.', h) # remove multiple continuous punctuation 
        h = re.sub('!+', '!', h)
        h = re.sub(r'\?+', '?', h)
        h = h.strip()

        # sentence tokenizeation can't process strange examples like "Jim decided to try his hand at a new task.repair a damaged oven.same with fixing a broken fridge.same with fixing a broken oven.same with fixing a broken oven.same with fixing a broken oven.same with fixing a broken oven.same with fixing a broken stove.sam"
        len_h = len(h)
        for pos in range(len_h):
            if h[pos] != '.':
                continue
            if pos == len(h) - 1:
                break
            if h[pos+1] != ' ':
                h = h[:pos+1]
                break

        h_sents = sent_tokenize(h)
        
        for h_sent in h_sents:
            punctuation = h_sent[-1]
            if full_sentence and punctuation not in ['.', '!']:
                continue
            if len(h_sent.split()) < 3:  # too short sentence such as "I did."
                continue

            if '~' in h_sent or '#' in h_sent or '…' in h_sent or '“' in h_sent:
                continue
            h_sent = h_sent.split('  ')[-1] # I.  I went to the store and bought more clothes.
            bleu_res = sacrebleu.sentence_bleu(h_sent, [input_left, input_right]) # discard repetitive one
            bleu_no_bp = bleu_res.score / bleu_res.bp
            if bleu_no_bp > bleu_threshold:
                continue
            return h_sent, logp

        return ''

    def bert_nsp_score(self, h, input_left, input_right):
        encoding = self.bert_tokenizer(input_left + ' ' + h, input_right, return_tensors='pt')
        encoding['input_ids'], encoding['token_type_ids'], encoding['attention_mask'] = encoding['input_ids'].cuda(), encoding['token_type_ids'].cuda(), encoding['attention_mask'].cuda()
        bert_outputs = self.bert(**encoding, labels=torch.LongTensor([1]).cuda())
        left_logits = bert_outputs.logits[0][0].data.cpu().numpy()
        encoding = self.bert_tokenizer(input_left, h + ' ' + input_right, return_tensors='pt')
        encoding['input_ids'], encoding['token_type_ids'], encoding['attention_mask'] = encoding['input_ids'].cuda(), encoding['token_type_ids'].cuda(), encoding['attention_mask'].cuda()
        bert_outputs = self.bert(**encoding, labels=torch.LongTensor([1]).cuda())
        right_logits = bert_outputs.logits[0][0].data.cpu().numpy()
        
        return left_logits + right_logits

    def predict_list(self, least_num=32, max_iter=8, need_logit=True, **kwargs):
        gen_hypos = []
        len_gen = 0
        iter_cnt = 0
        while len_gen <= least_num and iter_cnt <= max_iter:
            iter_cnt += 1
            _, gens = self.predict(need_fallback=False, **kwargs)
            torch.cuda.empty_cache()
            if gens[0] != '':
                gen_hypos += gens
                len_gen += len(gens)
        if len(gen_hypos) < least_num:
            gen_hypos += [('', -32767.0)] * least_num
        best_hypothesis = max(gen_hypos, key=lambda item: item[1])
        if not need_logit:
            gen_hypos = [text for text, _ in gen_hypos]
        return best_hypothesis, gen_hypos

    def beam_search(self, input_left='', input_right='', input_text=None, num_beams=5, num_return_sequences=5, top_p=0.9, min_length=7):
        with torch.no_grad():
            input_text = ' ' + input_left + ' <extra_id_0> ' + input_right if input_text is None else input_text
            inputs = self.tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=512)
            inputs["input_ids"] = inputs["input_ids"].cuda()
            inputs["attention_mask"] = inputs["attention_mask"].cuda()
            gens = self.model.generate(**inputs, do_sample=False, num_beams=num_beams, top_p=top_p, num_return_sequences=num_return_sequences, min_length=min_length, no_repeat_ngram_size=3)
            gen_text = self.tokenizer.batch_decode(gens, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            gen_text = [(gen, '') for gen in gen_text]
            gen_text = self.run_deberta_filter(input_left, input_right, gen_text, prob_threshold=0.0)
        return gen_text[0], gen_text

    def run_deberta_filter(self, input_left, input_right, outputs_valid, prob_threshold=0.9, dynamic_threshold=1.0, fallback=True):
        with torch.no_grad():
            pt_batch = self.deberta_tokenizer(
                [ [input_left + ' [SEP] ' + text, input_right] for text, _ in outputs_valid ],
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors="pt"
            )
            for k in pt_batch.keys():
                pt_batch[k] = pt_batch[k].cuda()
            logits = self.deberta_model(**pt_batch).logits
            torch.cuda.empty_cache()

        probs = self.softmax(logits)[:, 1-self.deberta_bad_label]
        outputs_with_prob = [tuple(output) + (probs[i].data.cpu().item(), ) for i, output in enumerate(outputs_valid)]
        # deberta_label = logits.argmax(dim=1)
        # outputs_valid = [output if deberta_label[i] != self.deberta_bad_label else '' for i, output in enumerate(outputs_valid)]
        outputs_valid = list(filter(lambda item: item[-1] >= prob_threshold, outputs_with_prob))
        if dynamic_threshold != 1.0 and outputs_valid != []:
            max_prob = probs.max().data.cpu().item()
            outputs_valid = list(filter(lambda item: item[-1] >= max_prob - dynamic_threshold, outputs_valid))
        if fallback and outputs_valid == []:
            outputs_valid = max(outputs_with_prob, key=lambda x: x[-1])
           
        return outputs_valid

    def run_deberta_filter_time_travel(self, premise, counterfactual, outputs_valid, prob_threshold=0.9, dynamic_threshold=1.0):
        with torch.no_grad():
            pt_batch = self.deberta_tokenizer(
                [ [premise + ' [SEP] ' + counterfactual, text] for text, _ in outputs_valid ],
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors="pt"
            )
            for k in pt_batch.keys():
                pt_batch[k] = pt_batch[k].cuda()
            logits = self.deberta_model(**pt_batch).logits
            torch.cuda.empty_cache()

        probs = self.softmax(logits)[:, 1-self.deberta_bad_label]
        outputs_valid = [tuple(output) + (probs[i].data.cpu().item(), ) for i, output in enumerate(outputs_valid)]
        # deberta_label = logits.argmax(dim=1)
        # outputs_valid = [output if deberta_label[i] != self.deberta_bad_label else '' for i, output in enumerate(outputs_valid)]
        outputs_valid = list(filter(lambda item: item[-1] >= prob_threshold, outputs_valid))
        if dynamic_threshold != 1.0 and outputs_valid != []:
            max_prob = probs.max().data.cpu().item()
            outputs_valid = list(filter(lambda item: item[-1] >= max_prob - dynamic_threshold, outputs_valid))
           
        return outputs_valid

    def run_bert_nsp_filter(self, input_left, input_right, outputs_valid):
        with torch.no_grad():
            whole_left_encoding, whole_right_encoding = \
                self.bert_tokenizer([(input_left + ' ' + h, input_right) for h, _ in outputs_valid], return_tensors="pt", padding=True), \
                    self.bert_tokenizer([(input_left, h + ' ' + input_right) for h, _ in outputs_valid], return_tensors="pt", padding=True)
            for e_key in ['input_ids', 'attention_mask', 'token_type_ids']:
                whole_left_encoding[e_key], whole_right_encoding[e_key] = \
                    whole_left_encoding[e_key].cuda(), whole_right_encoding[e_key].cuda()
            outputs_valid_length = len(outputs_valid)
            whole_left_logits, whole_right_logits = self.bert(**whole_left_encoding, labels=torch.LongTensor([[1] * outputs_valid_length]).cuda()).logits, \
                self.bert(**whole_right_encoding, labels=torch.LongTensor([[1] * outputs_valid_length]).cuda()).logits
            
        outputs_valid = [output if whole_left_logits[i][0] > whole_left_logits[i][1] and whole_right_logits[i][0] > whole_right_logits[i][1] else '' \
                for i, output in enumerate(outputs_valid)]
        outputs_valid = list(filter(lambda item: item != '', outputs_valid))
        torch.cuda.empty_cache()
        return outputs_valid

    def predict(self, input_left='', input_right='', length_lambda=0.0, topp=0.95, attempt_limit=3, batch_size=64, max_batch_size=256, max_input_length=128, max_output_length=32, min_output_length=7, \
        need_fallback=True, full_sentence=True, bert_nsp_filter=False, deberta_filter=False, bert_nsp_rerank=False, deberta_rerank=False, input_text=None, forbidden_output=None, prob_threshold=0.9):

        input_text = input_left + ' ' + self.infill_token + ' ' + input_right if input_text is None else input_text

        results, best_hypothesis = [], ''
        fall_back_results = []

        # tokenization
        with torch.no_grad():
            batch = self.tokenizer([input_text]*batch_size, return_tensors='pt', padding=True, max_length=max_input_length, truncation=True)
            for k in batch.keys():
                batch[k] = batch[k].cuda()

        for _ in range(attempt_limit):
            with torch.no_grad():
                # eos_token_id=self.extra_id_1_token if self.extra_id_1_token is not None else self.tokenizer.eos_token_id,
                generated_outputs = self.model.generate(**batch, \
                    max_length=max_output_length, do_sample=True, top_p=topp, num_return_sequences=4, \
                    output_scores=True, output_attentions=False, output_hidden_states=False, return_dict_in_generate=True, min_length=min_output_length)
                outputs_tokens = self.tokenizer.batch_decode(generated_outputs['sequences'], skip_special_tokens=True)

                # compute generation score
                probs = torch.stack(generated_outputs.scores, dim=1).log_softmax(-1)
                gen_probs = torch.gather(probs, -1, generated_outputs['sequences'][:, 1:, None]).squeeze(-1)
                gen_token_len = (gen_probs != float("-Inf")).count_nonzero(dim=1)
                # print(gen_token_len)
                gen_probs[gen_probs == float("-Inf")] = 0
                outputs_scores = gen_probs.sum(dim=1) / gen_token_len + length_lambda * gen_token_len
                # print(outputs_scores)
                outputs_flatten = list(set([(gen.replace('  ', ' '), score.cpu().data.detach().item()) for gen, score in zip(outputs_tokens, outputs_scores)]))
                # avoid CUDA out of memory. 
                del generated_outputs
                gc.collect()               
                torch.cuda.empty_cache()

                # outputs_valid = [self.valid_hypothesis(output, input_left, input_right, full_sentence) for output in outputs_flatten]
                if forbidden_output is None:
                    outputs_valid = list(filter(lambda item: item[0] != '' and item[0][-1] in ['.', '!'] and len(sent_tokenize(input_left + ' ' + item[0] + ' ' + input_right)) == 3, outputs_flatten))
                else:
                    outputs_valid = list(filter(lambda item: item[0] != forbidden_output, outputs_flatten))
                fall_back_results = outputs_valid

                if deberta_filter and outputs_valid != []:
                    outputs_valid = self.run_deberta_filter(input_left, input_right, outputs_valid, prob_threshold)


                if bert_nsp_filter and outputs_valid != []:
                    outputs_valid = self.run_bert_nsp_filter(input_left, input_right, outputs_valid)

                results += outputs_valid

            if results != []:
                break
            if batch_size < max_batch_size:
                batch_size *= 2 # everytime the model failed to infill, the batch_size doubles before reaching the max threshold.

        if need_fallback and len(results) == 0:
            results = fall_back_results[:256]
        if len(results) > 0:
            best_hypothesis = max(results, key=lambda item: item[1]) if deberta_rerank is False else max(results, key=lambda item: item[-1])
        else:
            best_hypothesis, results = ('', -32767.0), [('', -32767.0)]
        return best_hypothesis, results
