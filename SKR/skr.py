from __future__ import print_function
from collections import Counter
import string
import re
import json
import spacy

nlp = spacy.load('en_core_web_md')


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s.strip()))))


def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def exact_match_score(prediction, ground_truth):
    return int(normalize_answer(prediction) == normalize_answer(ground_truth))


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def get_answer_ref(prediction, answer):
    if 'answer is' in prediction:
        ans_index = prediction.index('answer is') + 9
        return prediction[ans_index:]
    else:
        nlp_text = nlp(prediction).sents
        sentences = [str(sent).strip() for sent in nlp_text]
        for sent in sentences:
            if answer.lower() in sent.lower():
                return answer
        return sentences[0].strip()


option = 'chain_of_thought_gpt3'

cot = []

test_path = '../results/cot_train.json'
test_file = open(test_path, 'r')
test_data = json.load(test_file)
f1 = exact_match = total = 0
for idx, test_case in enumerate(test_data):
    total += 1

    prediction = test_case[option]
    ground_truths = test_case['Gold answer']
    prediction = prediction.replace('\n', '').strip()
    prediction = get_answer_ref(prediction, ground_truths[0].strip())
    cot.append(metric_max_over_ground_truths(exact_match_score, prediction, ground_truths))
    exact_match += metric_max_over_ground_truths(
        exact_match_score, prediction, ground_truths)
    f1 += metric_max_over_ground_truths(
        f1_score, prediction, ground_truths)

print(100.0 * exact_match / total)
print(100.0 * f1 / total)

cot_ir = []

test_path = '../results/cot_train_ir.json'
test_file = open(test_path, 'r')
test_data2 = json.load(test_file)
f1 = exact_match = total = 0
for idx, test_case in enumerate(test_data2):
    total += 1
    prediction = test_case[option]
    ground_truths = test_case['Gold answer']
    prediction = prediction.replace('\n', '').strip()
    prediction = get_answer_ref(prediction, ground_truths[0].strip())
    cot_ir.append(metric_max_over_ground_truths(exact_match_score, prediction, ground_truths))
    exact_match += metric_max_over_ground_truths(
        exact_match_score, prediction, ground_truths)
    f1 += metric_max_over_ground_truths(
        f1_score, prediction, ground_truths)

print(100.0 * exact_match / total)
print(100.0 * f1 / total)

assert len(cot) == len(cot_ir)
assert len(test_data) == len(test_data2)

mixdata = []
ret = [0, 0, 0]
for i in range(len(cot)):
    tmp = {}
    tmp.update({'Question': test_data[i]['Question']})
    tmp.update({'Gold answer': test_data[i]['Gold answer']})
    tmp.update({'passages': test_data[i]['passages']})
    tmp.update({'cot': test_data[i]['chain_of_thought_gpt3']})
    tmp.update({'cot-ir': test_data2[i]['chain_of_thought_gpt3']})
    if cot[i] < cot_ir[i]:
        ret[0] += 1
        tmp.update({'status': 'ir better'})
    elif cot[i] > cot_ir[i]:
        ret[1] += 1
        tmp.update({'status': 'ir worse'})
    else:
        ret[2] += 1
        tmp.update({'status': 'same'})
    mixdata.append(tmp)

print(ret)

with open('./train_skr.json', 'w') as f:
    json.dump(mixdata, f, indent=4)
# [47, 62, 740]
