import pickle

import matplotlib.pyplot as plt


def plot_losses(losses, val_losses, save_path):
    plt.plot(losses, label="train_loss")
    plt.plot(val_losses, label="val_loss")
    plt.legend()
    plt.savefig(save_path)
    plt.show()


def load_losses(path_train, path_val):
    train_losses, val_losses = None, None
    with open(path_train, 'rb') as train_file:
        train_losses = pickle.load(train_file)
    with open(path_val, 'rb') as val_file:
        val_losses = pickle.load(val_file)
    return train_losses, val_losses


def main():
    experiment = 'ALBERT_LSTM'
    train_path = './experiment_results/' + experiment + '/losses.pkl'
    val_path = './experiment_results/' + experiment + '/val_losses.pkl'
    save_path = './experiment_results/' + experiment + '/losses.png'
    train, val = load_losses(train_path, val_path)
    plot_losses(train, val, save_path)


if __name__ == "__main__":
    main()
