import os
import sys
import json
import random
import glob
import argparse

import torch
import numpy as np

from seqeval.metrics import classification_report

from src.data.conll_dataset import Conll2003Dataset
from src.modules.bilstm_crf import BiLSTM_CRF
from src.modules.bilstm_fuzzycrf import BiLSTM_Fuzzy_CRF
from src.common.trainer import Trainer
from src.common.convert import convert

def main(file_name, output_file):
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

    if device == "GPU":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = "cpu"

    dataset = Conll2003Dataset(BATCH_SIZE,
                               DATASET_PATH,
                               word_emb_dim = WORD_EMBEDDING_DIM,
                               test_batch_size = 1,
                               pretrain_type = WORD_EMBEDDING_TYPE,
                               device = device)

    num_tags = dataset.num_tags
    label2idx, idx2labels = dataset.label2idx, dataset.idx2labels
    char2idx, idx2char = dataset.char2idx, dataset.idx2char
    word2idx, idx2word = dataset.word2idx, dataset.idx2word
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

    if device != "cpu":
        model = model.cuda(device)

    train_iter = dataset.train_batch
    valid_iter = dataset.valid_batch
    test_iter = dataset.test_batch

    with open(SAVE_MODEL_PATH + "/best_epoch.txt", "r") as f:
        best_epoch = f.read()

    best_model_name = "epoch_{}".format(best_epoch)

    model.load_state_dict(torch.load(SAVE_MODEL_PATH + "/" +best_model_name))
    model.eval()
    y_true, y_pred = [], []
    for batch in test_iter:
        predict_tags = model(batch)
        _, _, _, label_seq_tensor = batch
        y_true.extend(convert(label_seq_tensor.tolist(), label2idx))
        y_pred.extend(convert(predict_tags, label2idx))
    print(classification_report(y_true, y_pred, digits=5))
    
    if output_file is not None:
        with open(output_file, "w") as f:
            f.write("\n\n".join([ "\n".join(y_p) for y_p in y_pred]))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file")
    parser.add_argument("--output_file", default=None)
    args = parser.parse_args()
    main(args.input_file, args.output_file)