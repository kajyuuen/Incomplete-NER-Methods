import os
import sys
import json
import random

import torch
import numpy as np

from seqeval.metrics import classification_report

from src.data.conll_dataset import Conll2003Dataset
from src.modules.bilstm_crf import BiLSTM_CRF
from src.modules.bilstm_fuzzycrf import BiLSTM_Fuzzy_CRF
from src.modules.bilstm_hard_crf import BiLSTM_Hard_CRF
from src.common.trainer import Trainer
from src.common.hard_trainer import HardTrainer
from src.common.convert import convert

def main(file_name):
    with open(file_name, "r") as f:
        setting_json = json.load(f)

    # File path
    DATASET_PATH = setting_json["dataset_path"]
    SAVE_MODEL_PATH = setting_json["save_model_path"]

    # Word embedding
    WORD_EMBEDDING_TYPE = setting_json["model"]["embedding"]["word_embedding"]["type"]
    WORD_EMBEDDING_DIM = setting_json["model"]["embedding"]["word_embedding"]["dim"]

    # Char embedding
    CHAR_EMBEDDING_DIM = setting_json["model"]["embedding"]["char_embedding"]["dim"]
    CHAR_HIDDEN_DIM = setting_json["model"]["embedding"]["char_embedding"]["hidden_dim"]

    # Embedding encoder
    HIDDEN_DIM = setting_json["model"]["embedding"]["hidden_dim"]

    # Other
    MODEL_TYPE = setting_json["model"]["type"]
    DROPOUT_RATE = setting_json["model"]["dropout_rate"]
    if "seed" in setting_json:
        SEED = setting_json["seed"]
    else:
        SEED = 42

    # Use device
    device = setting_json["device"]

    # Train config
    LEAENING_RATE = setting_json["train"]["learning_rate"]
    EPOCHS = setting_json["train"]["epochs"]
    BATCH_SIZE = setting_json["train"]["batch_size"]

    if "clipping" in setting_json["train"]:
        CLIPPING = setting_json["train"]["clipping"]
    else:
        CLIPPING = None

    if "train_only" in setting_json["train"]:
        TRAIN_ONLY = setting_json["train"]["train_only"]
    else:
        TRAIN_ONLY = False

    if device == "GPU":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = "cpu"

    os.makedirs(SAVE_MODEL_PATH, exist_ok = True)
    with open(SAVE_MODEL_PATH + "/config.json", "w") as f:
        json.dump(setting_json, f)

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if device == "GPU":
        torch.cuda.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)

    dataset = Conll2003Dataset(BATCH_SIZE,
                               DATASET_PATH,
                               word_emb_dim = WORD_EMBEDDING_DIM,
                               pretrain_type = WORD_EMBEDDING_TYPE,
                               device = device)

    num_tags = dataset.num_tags
    label2idx, idx2labels = dataset.label2idx, dataset.idx2labels
    char2idx, idx2char = dataset.char2idx, dataset.idx2char
    word_embedding = dataset.word_embedding

    if MODEL_TYPE == "crf":
        model = BiLSTM_CRF(num_tags,
                        label2idx,
                        idx2labels,
                        word_embedding,
                        WORD_EMBEDDING_DIM,
                        HIDDEN_DIM,
                        CHAR_EMBEDDING_DIM,
                        CHAR_HIDDEN_DIM,
                        char2idx,
                        idx2char,
                        dropout_rate = DROPOUT_RATE,
                        device = device)
    elif MODEL_TYPE == "fuzzy_crf":
        model = BiLSTM_Fuzzy_CRF(num_tags,
                        label2idx,
                        idx2labels,
                        word_embedding,
                        WORD_EMBEDDING_DIM,
                        HIDDEN_DIM,
                        CHAR_EMBEDDING_DIM,
                        CHAR_HIDDEN_DIM,
                        char2idx,
                        idx2char,
                        dropout_rate = DROPOUT_RATE,
                        device = device)
    elif MODEL_TYPE == "hard_crf":
        model = BiLSTM_Hard_CRF(num_tags,
                        label2idx,
                        idx2labels,
                        word_embedding,
                        WORD_EMBEDDING_DIM,
                        HIDDEN_DIM,
                        CHAR_EMBEDDING_DIM,
                        CHAR_HIDDEN_DIM,
                        char2idx,
                        idx2char,
                        dropout_rate = DROPOUT_RATE,
                        device = device)

    if device != "cpu":
        model = model.cuda(device)
    
    train_iter = dataset.train_batch
    valid_iter = dataset.valid_batch
    test_iter = dataset.test_batch

    if MODEL_TYPE == "hard_crf":
        SUB_EPOCHS = setting_json["train"]["sub_num_epochs"]
        trainer = HardTrainer(model,
                        train_iter,
                        valid_iter=valid_iter,
                        test_iter=test_iter,
                        sub_num_epochs=SUB_EPOCHS,
                        label_dict=label2idx,
                        learning_rate=LEAENING_RATE,
                        clipping=CLIPPING,
                        save_path=SAVE_MODEL_PATH,
                        train_only=TRAIN_ONLY,
                        force_save=True,
                        device = device)
    else:
        trainer = Trainer(model,
                        train_iter,
                        valid_iter=valid_iter,
                        test_iter=test_iter,
                        label_dict=label2idx,
                        learning_rate=LEAENING_RATE,
                        clipping=CLIPPING,
                        save_path=SAVE_MODEL_PATH,
                        train_only=TRAIN_ONLY,
                        force_save=True)
    trainer.train(EPOCHS)

if __name__ == "__main__":
    file_name = sys.argv[1]
    main(file_name)