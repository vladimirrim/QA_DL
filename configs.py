class Config:
    def __init__(self):
        self.HIDDEN = 128
        self.BERT_MODEL = '/mnt/data/QA_DL/bert/'
        self.LOG_DIR = '/mnt/data/QA_DL/FFN_3/'
        self.TOKENIZER_PATH = '/mnt/data/QA_DL/bert/'
        self.DEV_DATASET = '/mnt/data/QA_DL/dev-v1.1.json'
        self.TRAIN_DATASET = '/mnt/data/QA_DL/train-v1.1.json'
        self.MAX_TEXT_LENGTH = 256
        self.DEVICE = 'cuda'
        self.LR = 0.00005
        self.BATCH_SIZE = 64
