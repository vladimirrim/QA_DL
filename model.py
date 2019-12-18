import torch
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
            conv_hidden_dims = [8, 16, 4]
        self.bert = BertModel.from_pretrained(config.BERT_MODEL)
        self.bert.eval()
        self.convLstm = ConvLSTM(input_size=(16, 16), input_dim=3, hidden_dim=conv_hidden_dims,
                                 kernel_size=(3, 3),
                                 num_layers=3,
                                 batch_first=True,
                                 bias=True,
                                 return_all_layers=False)
        self.flatten_for_qa = nn.Flatten(start_dim=2)
        self.qa_outputs = nn.Sequential(
            nn.Linear(conv_hidden_dims[-1] * 16 * 16, 512),
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
        output = output.reshape(output.shape[0], output.shape[1], 3, 16, 16)
        output = self.convLstm(output)
        output = output[0][0]
        sequence_output = self.flatten_for_qa(output)
        logits = self.qa_outputs(sequence_output)
        loss = None

        if start_positions is not None and end_positions is not None:
            loss = (self.loss_fct(logits[:, :, 0].masked_fill((1 - mask).bool(), float('-inf')), start_positions) + \
                    self.loss_fct(logits[:, :, 1].masked_fill((1 - mask).bool(), float('-inf')), end_positions)) / 2

        return loss, F.softmax(logits.masked_fill((1 - mask[:, :, None]).bool(), float('-inf')), dim=1)


class BertForQuestionAnsweringElmo(nn.Module):

    def __init__(self, config, hidden_lstm_dim=512):
        super().__init__()

        self.bert = BertModel.from_pretrained(config.BERT_MODEL)
        self.bert.eval()
        self.elmo_lstm = nn.LSTM(input_size=1024,
                                 hidden_size=hidden_lstm_dim,
                                 num_layers=2,
                                 batch_first=True,
                                 bidirectional=True)
        self.qa_outputs = nn.Sequential(nn.Linear(768 + 2 * hidden_lstm_dim, 512),
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

    def forward(self, elmo_emb, input_ids=None, token_type_ids=None, start_positions=None, end_positions=None,
                mask=None):
        output = self.bert(input_ids, attention_mask=mask)
        _, (h_n, _) = self.elmo_lstm(elmo_emb)
        fwd_final = h_n[0:h_n.size(0):2]  # [NumLayers, B, SrcEncoderH]
        bwd_final = h_n[1:h_n.size(0):2]  # [NumLayers, B, SrcEncoderH]
        h_n = torch.cat([fwd_final, bwd_final], dim=2)  # [NumLayers, B, NumDirections * SrcEncoderH]
        h_n = h_n[-1, :, :]

        sequence_output = output[0]
        sequence_output = torch.cat((sequence_output, h_n.unsqueeze(1).repeat(1, sequence_output.shape[1], 1)), dim=2)

        logits = self.qa_outputs(sequence_output)
        loss = None

        if start_positions is not None and end_positions is not None:
            loss = (self.loss_fct(logits[:, :, 0].masked_fill((1 - mask).bool(), float('-inf')), start_positions) + \
                    self.loss_fct(logits[:, :, 1].masked_fill((1 - mask).bool(), float('-inf')), end_positions)) / 2

        return loss, F.softmax(logits.masked_fill((1 - mask[:, :, None]).bool(), float('-inf')), dim=1)
