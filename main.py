import json
import os
import time

import torch
import pickle

from transformers import get_linear_schedule_with_warmup, AdamW

from configs import Config
from data_loader import DataLoader, get_text_question_ans_dataset
from evaluation import Evaluator
from model import BertForQuestionAnswering, BertForQuestionAnsweringConvLSTM


def initializeFolders(config):
    if not os.path.exists(config.LOG_DIR):
        os.makedirs(config.LOG_DIR)


if __name__ == '__main__':
    config = Config()
    dataLoader = DataLoader(config)
    initializeFolders(config)

    print('Starting loading data', flush=True)
    with open(config.TRAIN_DATASET, 'r') as train_json, open(config.DEV_DATASET, 'r') as dev_json:
        train_data = json.load(train_json)
        dev_data = json.load(dev_json)

    train_data_loader = dataLoader.get_data_loader(train_data)
    dev_data_loader = dataLoader.get_data_loader(dev_data)

    model = BertForQuestionAnsweringConvLSTM(config)
    model = model.to(config.DEVICE)

    epochs = 3
    t_total = len(train_data_loader) * epochs

    optimizer = AdamW(model.parameters(), lr=config.LR, eps=config.EPS, weight_decay=0.0)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=t_total)
    device = 'cuda'
    model.to(device)
    evaluator = Evaluator(model, dataLoader.tokenizer, config)
    start = time.time()
    losses = []
    val_losses = []
    best_val_loss = 10000
    print('Starting training', flush=True)
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for i, (texts, masks, start_pos, end_pos) in enumerate(train_data_loader):
            optimizer.zero_grad()
            loss, _ = model(texts.to(device),
                            mask=masks.to(device),
                            start_positions=torch.tensor(start_pos).to(device),
                            end_positions=torch.tensor(end_pos).to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.MAX_GRAD_NORM)
            train_loss += float(loss)
            optimizer.step()
            scheduler.step()
            if (i + 1) % 100 == 0:
                torch.save(model.state_dict(), config.LOG_DIR + 'bert.ckpt')
                print(f'Model saved on {i} iteration!', flush=True)
                end = time.time()
                print(f'Elapsed time: {end - start}', flush=True)
                start = end

                model.eval()
                val_loss = 0
                cnt = 0
                with torch.no_grad():
                    for texts, masks, start_pos, end_pos in dev_data_loader:
                        loss, _ = model(texts.to(device),
                                        mask=masks.to(device),
                                        start_positions=torch.tensor(start_pos).to(device),
                                        end_positions=torch.tensor(end_pos).to(device))
                        val_loss += float(loss)
                        cnt += 1
                val_losses.append(val_loss / cnt)
                losses.append(train_loss / 100)
                train_loss = 0
                print(f'Loss train: {losses[-1]}', flush=True)
                print(f'Loss dev: {val_losses[-1]}', flush=True)
                if best_val_loss > val_losses[-1]:
                    best_val_loss = val_losses[-1]
                    torch.save(model.state_dict(), config.LOG_DIR + 'bert_best_val.ckpt')
                    print(f'Best model on validation saved on {i} iteration!', flush=True)
                model.train()

    with open(config.LOG_DIR + 'losses.pkl', 'wb') as f:
        pickle.dump(losses, f)
    with open(config.LOG_DIR + 'val_losses.pkl', 'wb') as f:
        pickle.dump(val_losses, f)

    print(f'Starting evaluation', flush=True)
    dev_dataset = get_text_question_ans_dataset(dev_data)
    evaluator.evaluate(dev_dataset)
    print(f'Starting evaluation on best model on validation', flush=True)
    model.load_state_dict(torch.load(config.LOG_DIR + 'bert_best_val.ckpt'))
    evaluator.evaluate(dev_dataset)
