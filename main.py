import json
import os

import torch
import wandb

from configs import Config
from data_loader import DataLoader, get_text_question_ans_dataset
from evaluation import Evaluator
from model import BertForQuestionAnswering


def initializeFolders(config):
    if not os.path.exists(config.LOG_DIR):
        os.makedirs(config.LOG_DIR)


if __name__ == '__main__':
    config = Config()
    wandb.login(key='')
    wandb.init(dir=config.LOG_DIR)
    dataLoader = DataLoader(config)

    with open(config.TRAIN_DATASET, 'r') as train_json, open(config.DEV_DATASET, 'r') as dev_json:
        train_data = json.load(train_json)
        dev_data = json.load(dev_json)

    train_data_loader = dataLoader.get_data_loader(train_data)
    dev_dataset = get_text_question_ans_dataset(dev_data)

    model = BertForQuestionAnswering(config)
    model = model.to(config.DEVICE)

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), config.LR,
                                 weight_decay=0.000001)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.3)
    epochs = 3
    device = 'cuda'
    model.to(device)
    evaluator = Evaluator(model, dataLoader.tokenizer, config)
    for epoch in range(epochs):
        model.train()
        for i, (texts, masks, start_pos, end_pos) in enumerate(train_data_loader):
            optimizer.zero_grad()
            loss, _ = model(texts.to(device),
                            mask=masks.to(device),
                            start_positions=torch.tensor(start_pos).to(device),
                            end_positions=torch.tensor(end_pos).to(device))
            wandb.log({'loss': float(loss)})
            loss.backward()
            optimizer.step()
            if i % 100 == 0:
                print(f'Model saved on {i} iteration!')
                torch.save(model.state_dict(), './bert.pt')
                evaluator.evaluate(dev_dataset)
    evaluator.evaluate(dev_dataset)
