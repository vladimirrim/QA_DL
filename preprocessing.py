import pickle

import numpy as np


class Preprocessor:

    def __init__(self, config):
        self.MAX_TEXT_LEN = config.MAX_TEXT_LENGTH
        with open(config.TOKENIZER_PATH, 'rb') as f:
            self.tokenizer = pickle.load(f)

    def preprocessWindow(self, text_tokens, length, first, last):
        part_length = length // 3
        stride = 3 * part_length
        nrow = np.ceil(len(text_tokens) / part_length) - 2
        indexes = part_length * np.arange(nrow)[:, None] + np.arange(stride)
        indexes = indexes.astype(np.int32)

        max_index = indexes.max()
        diff = max_index + 1 - len(text_tokens)
        text_tokens += diff * [self.tokenizer.pad_token]
        text_tokens = list(np.array(text_tokens)[indexes])

        tokens = []
        labels = []
        for i, ts in enumerate(text_tokens):
            while ts[-1] == self.tokenizer.pad_token:
                ts = ts[:-1]

            tokens += [list(ts)]

            lfirst = first - i * part_length
            llast = last - i * part_length

            mask = 0 <= lfirst < len(ts) and 0 <= llast < len(ts)
            labels += [(lfirst if mask else 0, llast if mask else 0)]
        return tokens, labels

    def preprocess(self, text, question, answer):
        answer = answer.lower()
        if answer not in text.lower():
            return [], []

        firstInText = text.lower().find(answer)
        lastInText = firstInText + len(answer)
        text_tokens = self.tokenizer.tokenize(text[:firstInText].strip())
        first = len(text_tokens)
        text_tokens += self.tokenizer.tokenize(text[firstInText:lastInText].strip())
        last = len(text_tokens) - 1
        text_tokens += self.tokenizer.tokenize(text[lastInText:].strip())
        question_tokens = self.tokenizer.tokenize(question)

        length = self.MAX_TEXT_LEN - len(question_tokens) - 3
        if len(text_tokens) > length:
            tokens, labels = self.preprocessWindow(text_tokens, length, first, last)
        else:
            tokens = [text_tokens]
            labels = [(first, last)]

        for i in range(len(tokens)):
            tokens[i] = [self.tokenizer.cls_token] + \
                        question_tokens + \
                        [self.tokenizer.sep_token] + \
                        tokens[i] + \
                        [self.tokenizer.sep_token]
            labels[i] = (labels[i][0] + 2 + len(question_tokens), labels[i][1] + 2 + len(question_tokens))

        return tokens, labels
