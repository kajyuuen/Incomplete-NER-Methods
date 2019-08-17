import torch
import torch.nn as nn
import torch.nn.functional as F

from src.modules.char_bilstm import CharBiLSTM
from src.common.config import PAD_TAG, UNK_TAG
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class BiLSTM(nn.Module):
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
                 batch_first = True,
                 device = "cpu"):
        super().__init__()
        self.num_tags = num_tags
        self.idx2labels = idx2labels
        self.PAD_INDEX = label2idx[PAD_TAG]

        # Char embedding encoder
        self.char_hidden_dim = char_hidden_dim
        self.char_lstm = CharBiLSTM(char_emb_dim,
                                    self.char_hidden_dim,
                                    char2idx,
                                    idx2char,
                                    dropout = 0.5,
                                    device = device)

        # Word embedding encoder
        self.we_drop_layer = nn.Dropout(p = dropout_rate)
        self.word_emb = nn.Embedding.from_pretrained(torch.FloatTensor(word_embedding), freeze=False)

        # LSTM
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(word_emb_dim + self.char_hidden_dim,
                            self.hidden_dim // 2,
                            num_layers = 1,
                            batch_first = batch_first,
                            bidirectional = True)

        self.drop_layer = nn.Dropout(p = dropout_rate)
        self.hidden2tag = nn.Linear(self.hidden_dim, self.num_tags)

    def forward(self, batch):
        word_seq_tensor, word_seq_len, char_seq_tensor, char_seq_len, label_seq_tensor, possible_tags = batch
        batch_size, sequence_length = word_seq_tensor.size()

        # Get embedding
        # (batch, sequence_length, word_emb_dim)
        inputs_word_emb = self.word_emb(word_seq_tensor)
        # (batch, sequence_length, word_emb_dim)
        inputs_char_emb = self.char_lstm.get_last_hiddens(char_seq_tensor, char_seq_len)

        # Combine word and chat emb
        inputs_word_char_emb = self.we_drop_layer(torch.cat([inputs_word_emb, inputs_char_emb], 2))

        sorted_seq_len, perm_idx = word_seq_len.sort(0, descending=True)
        _, recover_idx = perm_idx.sort(0, descending=False)
        sorted_seq_tensor = inputs_word_char_emb[perm_idx]

        tags = label_seq_tensor

        # Create mask from tags
        # PAD_INDEX is 0, other index is 1
        mask = (tags != self.PAD_INDEX)

        # Convert embedding to feature
        lstm_out, _ = self.lstm(inputs_word_char_emb)
        lstm_out = self.drop_layer(lstm_out)
        feats = self.hidden2tag(lstm_out)

        return feats, tags, mask, possible_tags