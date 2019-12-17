class Config:
    def __init__(self):
        self.HIDDEN = 128
        self.BERT_MODEL = 'bert-base-multilingual-cased'
        self.LOG_DIR = '/mnt/data/QA_DL/LSTM/'
        self.TOKENIZER_PATH = '/mnt/data/QA_DL/tokenizer.pkl'
        self.DEV_DATASET = '/mnt/data/QA_DL/dev-v1.1.json'
        self.TRAIN_DATASET = '/mnt/data/QA_DL/train-v1.1.json'
        self.MAX_TEXT_LENGTH = 256
        self.DEVICE = 'cuda'
        self.LR = 0.00005
