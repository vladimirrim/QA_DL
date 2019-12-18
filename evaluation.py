import numpy as np
import torch

from data_loader import DataLoader, merge_to_words_all_sentences
from allennlp.modules.elmo import batch_to_ids

def calculate_intersection_len(a, b, c, d):
    left = max(a, c)
    right = min(b, d)
    return max(0, right - left + 1)


class Evaluator:
    def __init__(self, model, tokenizer, elmo, config) -> None:
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.data_loader = DataLoader(config, elmo)
        self.elmo = elmo

    def evaluate(self, dev_dataset):
        self.model.eval()
        total = 0
        correct = 0
        f1 = 0.0
        precision_total = 0.0
        recall_total = 0.0
        loss = 0
        total_loss = 0
        with torch.no_grad():
            for datapoint in dev_dataset:
                answer = datapoint[2].lower()
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

                        character_ids = batch_to_ids(merge_to_words_all_sentences([ts])).cuda()
                        embeddings = self.elmo(character_ids)['elmo_representations'][1]
                        texts, masks = self.data_loader.pad_sequence([ts])
                        texts = texts.to(self.config.DEVICE)
                        masks = masks.to(self.config.DEVICE)
                        lfirst = start_pos - i * part_length + 2 + len(question_tokens)
                        llast = end_pos - i * part_length + 2 + len(question_tokens)

                        mask = 0 <= lfirst < len(ts) and 0 <= llast < len(ts)
                        lfirst *= int(mask)
                        llast *= int(mask)
                        batch_loss, probs = self.model(embeddings, texts, mask=masks,
                            start_positions=torch.tensor(lfirst).view(-1).to(self.config.DEVICE),
                            end_positions=torch.tensor(llast).view(-1).to(self.config.DEVICE))
                        loss += float(batch_loss.cpu())
                        total_loss += 1

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
                    character_ids = batch_to_ids(merge_to_words_all_sentences([all_tokens])).cuda()
                    embeddings = self.elmo(character_ids)['elmo_representations'][1]
                    texts, masks = self.data_loader.pad_sequence([all_tokens])
                    texts = texts.to(self.config.DEVICE)
                    masks = masks.to(self.config.DEVICE)
                    batch_loss, probs = self.model(embeddings, texts, mask=masks,
                            start_positions=torch.tensor(start_pos + 2 + len(question_tokens)).view(-1).to(self.config.DEVICE),
                            end_positions=torch.tensor(end_pos + 2 + len(question_tokens)).view(-1).to(self.config.DEVICE))
                    loss += float(batch_loss.cpu())
                    total_loss += 1

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
        print(f'EM: {correct / total}', flush=True)
        print(f'F1: {f1 / total}', flush=True)
        print(f'Precision: {precision_total / total}', flush=True)
        print(f'Recall: {recall_total / total}', flush=True)
        print(f'Loss: {loss / total_loss}', flush=True)
        self.model.train()
        return loss / total_loss
