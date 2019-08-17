import torch
import torch.nn as nn
import torch.nn.functional as F

from src.modules.char_bilstm import CharBiLSTM
from src.modules.bilstm import BiLSTM
from src.modules.crf import CRF
from src.modules.fuzzy_crf import FuzzyCRF

from src.utils.math_utils import log_sum_exp
from src.utils.math_utils import argmax
from src.common.config import PAD_TAG, UNK_TAG


class BiLSTM_CRF(nn.Module):
    def __init__(self,
                 num_tags,
                 label2idx,
                 idx2labels,
                 word_embedding,
                 word_emb_dim,
                 hidden_dim,
                 char_emb_dim,
                 char_hidden_dim,
                 char2idx,
                 idx2char,
                 dropout_rate = 0,
                 word_vector = None,
                 batch_first = True,
                 inference = "CRF",
                 device = "cpu"):
        super().__init__()
        self.device = device
        self.encoder = BiLSTM(num_tags,
                              label2idx,
                              idx2labels,
                              word_embedding,
                              word_emb_dim,
                              hidden_dim,
                              char_emb_dim,
                              char_hidden_dim,
                              char2idx,
                              idx2char,
                              batch_first = batch_first,
                              device = device)
        if inference == "CRF":
            self.inferencer = CRF(num_tags,
                                label2idx,
                                idx2labels,
                                device = device)
        elif inference == "FuzzyCRF":
            self.inferencer = FuzzyCRF(num_tags,
                                label2idx,
                                idx2labels,
                                device = device)

    
    def forward(self, batch):
        feats, tags, mask = self.encoder(batch)
        score = self.inferencer(feats, tags, mask)
        return score

    def decode(self, batch):
        feats, _, mask = self.encoder(batch)
        return self.inferencer._viterbi_tags(feats, mask)