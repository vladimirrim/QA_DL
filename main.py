import glob
import re

import torch
from transformers import BertTokenizer, BertForQuestionAnswering


class Dataset:
    def __init__(self):
        self.story = ""
        self.stories = []
        self.queries = []
        self.answers = []

    def preprocessLine(self, line):
        tokens = line.split()
        story = self.story
        if tokens[0] == '1':
            self.story = ""
        if '?' in line:
            self.answers.append(tokens[-3])
            self.queries.append(' '.join(tokens[1:-3]))
            self.stories.append(story)
        else:
            self.story += ' ' + ' '.join(tokens[1:])


if __name__ == '__main__':
    prefix = '.data/tasks_1-20_v1-2/en-valid-10k/'
    nlp = 'dataset_281937_1.txt'
    test, train = Dataset(), Dataset()
    for filename in glob.glob(prefix + '*test.txt'):
        with open(filename) as f:
            [test.preprocessLine(line) for line in f.readlines()]

    for filename in glob.glob(prefix + '*train.txt'):
        with open(filename) as f:
            [train.preprocessLine(line) for line in f.readlines()]
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')

    total = 0
    tp = 0
    for q, a, s in zip(test.queries, test.answers, test.stories):
        question, text = q, s
        input_text = "[CLS] " + question + " [SEP] " + text + " [SEP]"
        input_ids = tokenizer.encode(input_text)
        token_type_ids = [0 if i <= input_ids.index(102) else 1 for i in range(len(input_ids))]
        start_scores, end_scores = model(torch.tensor([input_ids]), token_type_ids=torch.tensor([token_type_ids]))
        all_tokens = tokenizer.convert_ids_to_tokens(input_ids)
        predict = re.sub(' ##', '', ' '.join(all_tokens[torch.argmax(start_scores): torch.argmax(end_scores) + 1])).replace('the', '')
        tp += int(predict.replace(' ', '') == a)
        total += 1
    print(tp / total)



