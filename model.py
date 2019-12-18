from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import BertModel

import torch.nn.functional as F


class BertForQuestionAnswering(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.bert = BertModel.from_pretrained(config.BERT_MODEL)
        self.bert.eval()
        self.qa_outputs = nn.Sequential(nn.Linear(768, 512),
                                        nn.ReLU(),
                                        nn.Dropout(p=0.5),
                                        nn.Linear(512, 128),
                                        nn.ReLU(),
                                        nn.Dropout(p=0.2),
                                        nn.Linear(128, 64),
                                        nn.ReLU(),
                                        nn.Dropout(p=0.1),
                                        nn.Linear(64, 2))
        self.loss_fct = CrossEntropyLoss()

    def forward(self, input_ids=None, token_type_ids=None, start_positions=None, end_positions=None, mask=None):
        output = self.bert(input_ids, attention_mask=mask)

        sequence_output = output[0]

        logits = self.qa_outputs(sequence_output)
        loss = None

        if start_positions is not None and end_positions is not None:
            loss = (self.loss_fct(logits[:, :, 0].masked_fill((1 - mask).bool(), float('-inf')), start_positions) + \
                    self.loss_fct(logits[:, :, 1].masked_fill((1 - mask).bool(), float('-inf')), end_positions)) / 2

        return loss, F.softmax(logits.masked_fill((1 - mask[:, :, None]).bool(), float('-inf')), dim=1)
