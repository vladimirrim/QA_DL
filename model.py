from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import BertModel

import torch.nn.functional as F

from ConvLSTM import ConvLSTM


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


class BertForQuestionAnsweringLSTM(nn.Module):

    def __init__(self, config, hidden_lstm_dim=512):
        super().__init__()
        self.bert = BertModel.from_pretrained(config.BERT_MODEL)
        self.bert.eval()
        self.lstm = nn.LSTM(input_size=768,
                            hidden_size=hidden_lstm_dim,
                            num_layers=2,
                            batch_first=True,
                            bidirectional=True)
        self.qa_outputs = nn.Sequential(
            nn.Linear(hidden_lstm_dim * 2, 512),
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

        output = output[0]
        output, _ = self.lstm(output)
        logits = self.qa_outputs(output)
        loss = None
        if start_positions is not None and end_positions is not None:
            loss = (self.loss_fct(logits[:, :, 0].masked_fill((1 - mask).bool(), float('-inf')), start_positions) + \
                    self.loss_fct(logits[:, :, 1].masked_fill((1 - mask).bool(), float('-inf')), end_positions)) / 2

        return loss, F.softmax(logits.masked_fill((1 - mask[:, :, None]).bool(), float('-inf')), dim=1)

class BertForQuestionAnsweringConvLSTM(nn.Module):

    def __init__(self, config, conv_hidden_dims=None):
        super().__init__()
        if conv_hidden_dims is None:
            conv_hidden_dims = [4, 16, 4]
        self.bert = BertModel.from_pretrained(config.BERT_MODEL)
        self.bert.eval()
        self.convLstm = ConvLSTM(input_size=(1, 768), input_dim=1, hidden_dim=conv_hidden_dims,
                                       kernel_size=(1, 15),
                                       num_layers=3,
                                       batch_first=True,
                                       bias=True,
                                       return_all_layers=False)
        self.flatten_for_qa = nn.Flatten(start_dim=2)
        self.qa_outputs = nn.Sequential(
            nn.Linear(conv_hidden_dims[-1] * 768, 512),
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

        output = output[0]
        output = output.unsqueeze(2).unsqueeze(3)
        output = self.convLstm(output)
        output = output[0][0]
        sequence_output = self.flatten_for_qa(output)
        logits = self.qa_outputs(sequence_output)
        loss = None

        if start_positions is not None and end_positions is not None:
            loss = (self.loss_fct(logits[:, :, 0].masked_fill((1 - mask).bool(), float('-inf')), start_positions) + \
                    self.loss_fct(logits[:, :, 1].masked_fill((1 - mask).bool(), float('-inf')), end_positions)) / 2

        return loss, F.softmax(logits.masked_fill((1 - mask[:, :, None]).bool(), float('-inf')), dim=1)
