class Config:
    def __init__(self):
        self.HIDDEN = 128
        self.BERT_MODEL = 'bert-base-multilingual-cased'
        self.LOG_DIR = '/mnt/data/QA_DL/LSTM/'
        self.DEV_DATASET = './gdrive/My Drive/datasets_for_homeworks/dev-v1.1.json'
        self.TRAIN_DATASET = './gdrive/My Drive/datasets_for_homeworks/train-v1.1.json'
        self.MAX_TEXT_LENGTH = 256
        self.DEVICE = 'cuda'
        self.LR = 0.00005
