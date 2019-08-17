PAD_TAG = "<pad>"
PAD_ID = 0
UNK_TAG = "<unk>"
UNK_ID = 1
UNLABELED_TAG = "NOANNOTATION"
UNLABELED_ID = -1

import json

import torch

class Config:

    def __init__(self, setting_json):
        self.SEED = 42

        # File path
        self.DATASET_PATH = setting_json["dataset_path"]
        self.SAVE_MODEL_PATH = setting_json["save_model_path"]

        # Word embedding
        self.WORD_EMBEDDING_TYPE = setting_json["model"]["embedding"]["word_embedding"]["type"]
        self.WORD_EMBEDDING_DIM = setting_json["model"]["embedding"]["word_embedding"]["dim"]

        # Char embedding
        self.CHAR_EMBEDDING_DIM = setting_json["model"]["embedding"]["char_embedding"]["dim"]
        self.CHAR_HIDDEN_DIM = setting_json["model"]["embedding"]["char_embedding"]["hidden_dim"]

        # Embedding encoder
        self.HIDDEN_DIM = setting_json["model"]["embedding"]["hidden_dim"]

        # Other
        self.MODEL_TYPE = setting_json["model"]["type"]
        self.DROPOUT_RATE = setting_json["model"]["dropout_rate"]
        if "seed" in setting_json:
            SEED = setting_json["seed"]
        else:
            SEED = 42

        # Use device
        self.device = setting_json["device"]

        # Train config
        self.LEAENING_RATE = setting_json["train"]["learning_rate"]
        self.EPOCHS = setting_json["train"]["epochs"]
        self.BATCH_SIZE = setting_json["train"]["batch_size"]

        if "clipping" in setting_json["train"]:
            self.CLIPPING = setting_json["train"]["clipping"]
        else:
            self.CLIPPING = None

        if "train_only" in setting_json["train"]:
            self.TRAIN_ONLY = setting_json["train"]["train_only"]
        else:
            self.TRAIN_ONLY = False

        if self.device == "GPU":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = "cpu"

        if self.MODEL_TYPE == "simple":
            self.unlabel_to_other = True
        else:
            self.unlabel_to_other = False
