import pickle

import torch
from allennlp.modules.elmo import batch_to_ids

from transformers import BertTokenizer
from preprocessing import Preprocessor


def get_text_question_ans_dataset(squad_dataset):
    tqa_dataset = []
    for d in squad_dataset['data']:
        for p in d['paragraphs']:
            for qa in p['qas']:
                # TODO: deal with several answers
                tqa_dataset.append((p['context'], qa['question'], qa['answers'][0]['text']))
    return tqa_dataset


def convert_to_words(tokens):
    result = []
    for token in tokens:
        if token.startswith('##'):
            result[-1] += token[2:]
        else:
            result.append(token)
    return result


def merge_to_words_all_sentences(d_tokens):
    result = []
    for tokens in d_tokens:
        result.append(convert_to_words(tokens))
    return result


class DataLoader:
    def __init__(self, config, elmo):
        self.tokenizer = BertTokenizer.from_pretrained(config.BERT_MODEL)
        self.preprocessor = Preprocessor(config)
        self.elmo = elmo

    def pad_sequence(self, texts):
        max_len = max([len(text) for text in texts])
        masks = [[1] * len(text) + [0] * (max_len - len(text)) for text in texts]
        texts = [text + [self.tokenizer.pad_token] * (max_len - len(text)) for text in texts]
        texts = [self.tokenizer.convert_tokens_to_ids(text) for text in texts]
        texts = torch.LongTensor(texts)
        masks = torch.LongTensor(masks)

        return texts, masks

    def collate_fn(self, data):
        texts, labels, elmo_texts = zip(*data)
        texts, masks = self.pad_sequence(texts)
        character_ids = batch_to_ids(elmo_texts).cuda()
        embeddings = self.elmo(character_ids)

        labels_first, labels_last = zip(*labels)
        start_pos = labels_first
        end_pos = labels_last
        return embeddings['elmo_representations'][1], texts, masks, torch.LongTensor(start_pos), torch.LongTensor(end_pos)

    def get_data_loader(self, squad_dataset):
        dataset_tokens, dataset_labels = [], []
        tqa_train_dataset = get_text_question_ans_dataset(squad_dataset)
        for datapoint in tqa_train_dataset:
            tokens, labels = self.preprocessor.preprocess(datapoint[0], datapoint[1], datapoint[2])
            dataset_tokens += tokens
            dataset_labels += labels
        dataset_tokens_elmo = merge_to_words_all_sentences(dataset_tokens)
        return torch.utils.data.DataLoader(list(zip(dataset_tokens, dataset_labels, dataset_tokens_elmo)), batch_size=self.config.BATCH_SIZE,
                                           shuffle=True, collate_fn=self.collate_fn)
