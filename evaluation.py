import numpy as np
import torch

from data_loader import DataLoader


def calculate_intersection_len(a, b, c, d):
    left = max(a, c)
    right = min(b, d)
    return max(0, right - left + 1)


class Evaluator:
    def __init__(self, model, tokenizer, config) -> None:
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.data_loader = DataLoader(config)

    def evaluate(self, dev_dataset):
        self.model.eval()
        total = 0
        correct = 0
        f1 = 0.0
        precision_total = 0.0
        recall_total = 0.0
        with torch.no_grad():
            for datapoint in dev_dataset:
                answer = datapoint[3].lower()
                if answer not in datapoint[0].lower():
                    continue
                total += 1
                firstInText = datapoint[0].lower().find(answer)
                lastInText = firstInText + len(answer)
                text_tokens = self.tokenizer.tokenize(datapoint[0][:firstInText].strip())
                start_pos = len(text_tokens)
                text_tokens += self.tokenizer.tokenize(datapoint[0][firstInText:lastInText].strip())
                end_pos = len(text_tokens) - 1
                text_tokens += self.tokenizer.tokenize(datapoint[0][lastInText:].strip())
                question_tokens = self.tokenizer.tokenize(datapoint[1])

                all_tokens = [self.tokenizer.cls_token] + \
                             question_tokens + \
                             [self.tokenizer.sep_token] + \
                             text_tokens + \
                             [self.tokenizer.sep_token]

                length = self.config.MAX_TEXT_LENGTH - len(question_tokens) - 3
                if len(text_tokens) > length:
                    part_length = length // 3
                    stride = 3 * part_length
                    nrow = np.ceil(len(text_tokens) / part_length) - 2
                    indexes = part_length * np.arange(nrow)[:, None] + np.arange(stride)
                    indexes = indexes.astype(np.int32)

                    max_index = indexes.max()
                    diff = max_index + 1 - len(text_tokens)
                    text_tokens += diff * [self.tokenizer.pad_token]

                    text_tokens = np.array(text_tokens)[indexes].tolist()

                    start, end, prob = 0, 0, 0
                    for i, ts in enumerate(text_tokens):
                        while ts[-1] == self.tokenizer.pad_token:
                            ts = ts[:-1]

                        ts = [self.tokenizer.cls_token] + \
                             question_tokens + \
                             [self.tokenizer.sep_token] + \
                             ts + \
                             [self.tokenizer.sep_token]

                        texts, masks = self.data_loader.pad_sequence([ts])
                        texts = texts.to(self.config.DEVICE)
                        masks = masks.to(self.config.DEVICE)

                        probs = self.model(texts, mask=masks)[1]

                        start_max = torch.argmax(probs[0, :, 0])
                        end_max = torch.argmax(probs[0, :, 1])
                        new_prob = probs[0][start_max][0] * probs[0][start_max][1]
                        if new_prob > prob:
                            prob = new_prob
                            start, end = torch.min(start_max, end_max), torch.max(start_max, end_max)

                    start_pos_true = start_pos + 2 + len(question_tokens)
                    end_pos_true = end_pos + 2 + len(question_tokens)

                    first = start_pos_true == start
                    second = end_pos_true == end
                    correct += int(first and second)

                    intersection_len = calculate_intersection_len(start, end, start_pos_true, end_pos_true)
                    precision = intersection_len / (end - start + 1)
                    recall = intersection_len / (end_pos_true - start_pos_true + 1)

                    precision_total += precision
                    recall_total += recall
                    if precision + recall != 0:
                        f1 += 2 * precision * recall / (precision + recall)
                else:
                    texts, masks = self.data_loader.pad_sequence([all_tokens])
                    texts = texts.to(self.config.device)
                    masks = masks.to(self.config.device)
                    probs = self.model(texts, mask=masks)[1]

                    start_max = torch.argmax(probs[0, :, 0])
                    end_max = torch.argmax(probs[0, :, 1])
                    start, end = torch.min(start_max, end_max), torch.max(start_max, end_max)
                    start_pos_true = start_pos + 2 + len(question_tokens)
                    end_pos_true = end_pos + 2 + len(question_tokens)

                    first = start_pos_true == start
                    second = end_pos_true == end
                    correct += int(first and second)

                    intersection_len = calculate_intersection_len(start, end, start_pos_true, end_pos_true)
                    precision = intersection_len / (end - start + 1)
                    recall = intersection_len / (end_pos_true - start_pos_true + 1)

                    precision_total += precision
                    recall_total += recall
                    if precision + recall != 0:
                        f1 += 2 * precision * recall / (precision + recall)
        print(f'EM: {correct / total}')
        print(f'F1: {f1 / total}')
        print(f'Precision: {precision_total / total}')
        print(f'Recall: {recall_total / total}')
        return correct / total, f1 / total, precision_total / total, recall_total / total