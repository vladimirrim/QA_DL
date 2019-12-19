import os
import pickle
from typing import List

import matplotlib.pyplot as plt


def plot_losses(losses: List[List[float]], labels: List[str]) -> None:
    plt.title("Loss")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    for loss_values, label in zip(losses, labels):
        plt.plot([i * 100 for i in range(len(loss_values))], loss_values, label=label)
        plt.legend()
    plt.show()


def draw_plots():
    with open(os.path.join('../QA_data/ELMO_FFN_QA_1/losses.pkl'), 'rb') as train_file:
        train_losses = pickle.load(train_file)
    with open(os.path.join('../QA_data/ELMO_FFN_QA_1/val_losses.pkl'), 'rb') as val_file:
        val_losses = pickle.load(val_file)
    plot_losses([train_losses, val_losses], ['train', 'validation'])


if __name__ == "__main__":
    draw_plots()