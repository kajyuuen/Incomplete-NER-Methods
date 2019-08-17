import os
import sys
import json
import random

import torch
import numpy as np

from seqeval.metrics import classification_report

from src.data.conll_dataset import Conll2003Dataset
from src.modules.bilstm_crf import BiLSTM_CRF
from src.modules.bilstm_hard_crf import BiLSTM_Hard_CRF

from src.common.config import Config
from src.common.trainer import Trainer
from src.common.hard_trainer import HardTrainer
from src.common.convert import convert

def main(file_name):
    with open(file_name, "r") as f:
        setting_json = json.load(f)
    config = Config(setting_json)

    os.makedirs(config.SAVE_MODEL_PATH, exist_ok = True)
    with open(config.SAVE_MODEL_PATH + "/config.json", "w") as f:
        json.dump(setting_json, f)

    random.seed(config.SEED)
    np.random.seed(config.SEED)
    torch.manual_seed(config.SEED)
    if config.device == "GPU":
        torch.cuda.manual_seed(config.SEED)
        torch.cuda.manual_seed_all(config.SEED)

    dataset = Conll2003Dataset(config.BATCH_SIZE,
                               config.DATASET_PATH,
                               word_emb_dim = config.WORD_EMBEDDING_DIM,
                               pretrain_type = config.WORD_EMBEDDING_TYPE,
                               unlabel_to_other = config.unlabel_to_other,
                               device = config.device)

    num_tags = dataset.num_tags
    label2idx, idx2labels = dataset.label2idx, dataset.idx2labels
    char2idx, idx2char = dataset.char2idx, dataset.idx2char
    word_embedding = dataset.word_embedding

    if config.MODEL_TYPE == "crf" or config.MODEL_TYPE == "simple":
        model = BiLSTM_CRF(num_tags,
                        label2idx,
                        idx2labels,
                        word_embedding,
                        config.WORD_EMBEDDING_DIM,
                        config.HIDDEN_DIM,
                        config.CHAR_EMBEDDING_DIM,
                        config.CHAR_HIDDEN_DIM,
                        char2idx,
                        idx2char,
                        dropout_rate = config.DROPOUT_RATE,
                        device = config.device)
    elif config.MODEL_TYPE == "fuzzy_crf":
        model = BiLSTM_CRF(num_tags,
                        label2idx,
                        idx2labels,
                        word_embedding,
                        config.WORD_EMBEDDING_DIM,
                        config.HIDDEN_DIM,
                        config.CHAR_EMBEDDING_DIM,
                        config.CHAR_HIDDEN_DIM,
                        char2idx,
                        idx2char,
                        dropout_rate = config.DROPOUT_RATE,
                        inference = "FuzzyCRF",
                        device = config.device)
    elif config.MODEL_TYPE == "hard_crf":
        model = BiLSTM_Hard_CRF(num_tags,
                        label2idx,
                        idx2labels,
                        word_embedding,
                        config.WORD_EMBEDDING_DIM,
                        config.HIDDEN_DIM,
                        config.CHAR_EMBEDDING_DIM,
                        config.CHAR_HIDDEN_DIM,
                        char2idx,
                        idx2char,
                        dropout_rate = config.DROPOUT_RATE,
                        device = config.device)

    if config.device != "cpu":
        model = model.cuda(config.device)
    
    train_iter = dataset.train_batch
    valid_iter = dataset.valid_batch
    test_iter = dataset.test_batch

    if config.MODEL_TYPE == "hard_crf":
        SUB_EPOCHS = setting_json["train"]["sub_num_epochs"]
        trainer = HardTrainer(model,
                        train_iter,
                        valid_iter=valid_iter,
                        test_iter=test_iter,
                        sub_num_epochs=SUB_EPOCHS,
                        label_dict=label2idx,
                        learning_rate=config.LEAENING_RATE,
                        clipping=config.CLIPPING,
                        save_path=config.SAVE_MODEL_PATH,
                        train_only=config.TRAIN_ONLY,
                        force_save=True,
                        device=config.device)
    else:
        trainer = Trainer(model,
                        train_iter,
                        valid_iter=valid_iter,
                        test_iter=test_iter,
                        label_dict=label2idx,
                        learning_rate=config.LEAENING_RATE,
                        clipping=config.CLIPPING,
                        save_path=config.SAVE_MODEL_PATH,
                        train_only=config.TRAIN_ONLY,
                        force_save=True)
    trainer.train(config.EPOCHS)

if __name__ == "__main__":
    file_name = sys.argv[1]
    main(file_name)