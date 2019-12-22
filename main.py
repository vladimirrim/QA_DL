import json
import os
import pickle
import time

import torch

from configs import Config
from data_loader import DataLoader, get_text_question_ans_dataset
from elmo import create_elmo
from evaluation import Evaluator
from model import BertForQuestionAnsweringElmoLSTM


def initializeFolders(config):
    if not os.path.exists(config.LOG_DIR):
        os.makedirs(config.LOG_DIR)


if __name__ == '__main__':
    config = Config()
    elmo = create_elmo(config)
    dataLoader = DataLoader(config, elmo)
    initializeFolders(config)

    print('Starting loading data', flush=True)
    with open(config.TRAIN_DATASET, 'r') as train_json, open(config.DEV_DATASET, 'r') as dev_json:
        train_data = json.load(train_json)
        dev_data = json.load(dev_json)

    train_data_loader = dataLoader.get_data_loader(train_data)
    dev_data_loader = dataLoader.get_data_loader(dev_data)

    model = BertForQuestionAnsweringElmoLSTM(config)
    model = model.to(config.DEVICE)

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), config.LR,
                                 weight_decay=0.000001)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.3)
    epochs = 3
    device = 'cuda'
    model.to(device)
    evaluator = Evaluator(model, dataLoader.tokenizer, elmo, config)
    start = time.time()
    losses = []
    val_losses = []
    best_val_loss = 10000
    print('Starting training', flush=True)
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for i, (elmo_emb, texts, masks, start_pos, end_pos) in enumerate(train_data_loader):
            optimizer.zero_grad()
            loss, _ = model(elmo_emb.to(device),
                            texts.to(device),
                            mask=masks.to(device),
                            start_positions=torch.tensor(start_pos).to(device),
                            end_positions=torch.tensor(end_pos).to(device))
            loss.backward()
            train_loss += float(loss)
            optimizer.step()
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
                    for elmo_emb, texts, masks, start_pos, end_pos in dev_data_loader:
                        loss, _ = model(elmo_emb.to(device),
                                        texts.to(device),
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
