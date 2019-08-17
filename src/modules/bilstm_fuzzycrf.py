import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.modules.char_bilstm import CharBiLSTM
from src.utils.math_utils import log_sum_exp
from src.utils.math_utils import argmax
from src.common.config import PAD_TAG, UNK_TAG, UNLABELED_TAG

def possible_tag_masks(num_tags, tags, unlabeled_index, device = "cpu"):
    no_annotation_idx = (tags == unlabeled_index)
    tags[tags == unlabeled_index] = 0

    tags_ = torch.unsqueeze(tags, 2)
    masks = torch.zeros(tags_.size(0), tags_.size(1), num_tags, device=device)
    masks.scatter_(2, tags_, 1)
    masks[no_annotation_idx] = 1
    return masks

class BiLSTM_Fuzzy_CRF(nn.Module):
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
        self.UNLABELED_INDEX = label2idx[UNLABELED_TAG]
        self.device = device
        print("BiLSTM_Fuzzy_CRF")
        # Char embedding encoder
        self.char_hidden_dim = char_hidden_dim
        self.char_lstm = CharBiLSTM(char_emb_dim,
                                    self.char_hidden_dim,
                                    char2idx,
                                    idx2char,
                                    dropout = 0.5,
                                    device=self.device)

        # Word embedding encoder
        self.we_drop_layer = nn.Dropout(p = dropout_rate)
        self.word_emb = nn.Embedding.from_pretrained(torch.FloatTensor(word_embedding), freeze=False).to(self.device)

        # LSTM
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(word_emb_dim + self.char_hidden_dim,
                            self.hidden_dim // 2,
                            num_layers = 1,
                            batch_first = batch_first,
                            bidirectional = True)

        self.drop_layer = nn.Dropout(p = dropout_rate)
        self.hidden2tag = nn.Linear(self.hidden_dim, self.num_tags)
 
        # CRF
        init_transition = torch.randn(self.num_tags, self.num_tags)
        init_transition[:, self.PAD_INDEX] = -10000.0
        init_transition[self.PAD_INDEX, :] = -10000.0

        self.transitions = nn.Parameter(init_transition)
        self.start_transitions = torch.nn.Parameter(torch.Tensor(self.num_tags))
        self.end_transitions = torch.nn.Parameter(torch.Tensor(self.num_tags))

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_normal_(self.transitions)
        torch.nn.init.normal_(self.start_transitions)
        torch.nn.init.normal_(self.end_transitions)
    
    def _forward_alg(self, feats, mask):
        """
        Parameters:
            feats: (batch_size, sequence_length, num_tags)
            mask: Show padding tags. 0 don't calculate score. (batch_size, sequence_length)
        Returns:
            scores: (batch_size)
        """
        batch_size, sequence_length, num_tags = feats.data.shape

        feats = feats.transpose(0, 1).contiguous()
        mask = mask.float().transpose(0, 1).contiguous()

        # Start transition score and first feats
        alpha = self.start_transitions.view(1, num_tags) + feats[0]

        for i in range(1, sequence_length):

            feats_score = feats[i].view(batch_size, 1, num_tags)             # (batch_size, 1, num_tags)
            transition_scores = self.transitions.view(1, num_tags, num_tags) # (batch_size, num_tags, num_tags)
            broadcast_alpha = alpha.view(batch_size, num_tags, 1)

            inner = broadcast_alpha + feats_score + transition_scores        # (batch_size, num_tags)

            alpha = (log_sum_exp(inner, 1) * mask[i].view(batch_size, 1) +
                     alpha * (1 - mask[i]).view(batch_size, 1))

        # Add end transition score
        stops = alpha + self.end_transitions.view(1, num_tags)

        # Sum (log-sum-exp) over all possible tags
        return log_sum_exp(stops) # (batch_size,)

    def _score_sentence(self, feats, tags, mask):
        """
        Parameters:
            feats: (batch_size, sequence_length, num_tags)
            tags:  (batch_size, sequence_length)
            mask:  Show padding tags. 0 don't calculate score. (batch_size, sequence_length)
        Returns:
            scores: (batch_size)
        """

        batch_size, sequence_length, num_tags = feats.data.shape

        feats = feats.transpose(0, 1).contiguous()
        _tags = copy.deepcopy(tags).contiguous()
        mask = mask.float().transpose(0, 1).contiguous()

        possible_tags = possible_tag_masks(self.num_tags, _tags, self.UNLABELED_INDEX, self.device).float().transpose(0, 1) # (sequence_length, batch_size, num_tags)

        # Start transition score and first emission
        first_possible_tag = possible_tags[0]

        alpha = self.start_transitions + feats[0]      # (batch_size, num_tags)
        alpha[(first_possible_tag == 0)] = -10000000

        for i in range(1, sequence_length):
            current_possible_tags = possible_tags[i-1] # (batch_size, num_tags)
            next_possible_tags = possible_tags[i]      # (batch_size, num_tags)

            # The emit scores are for time i ("next_tag") so we broadcast along the current_tag axis.
            feats_score = feats[i]
            feats_score[(next_possible_tags == 0)] = -10000000
            feats_score = feats_score.view(batch_size, 1, num_tags)

            # Transition scores are (current_tag, next_tag) so we broadcast along the instance axis.
            transition_scores = self.transitions.view(1, num_tags, num_tags).expand(batch_size, num_tags, num_tags) \
                        * current_possible_tags.unsqueeze(-1).expand(batch_size, num_tags, num_tags) \
                        * next_possible_tags.unsqueeze(-1).expand(batch_size, num_tags, num_tags).transpose(1, 2)
            transition_scores[(transition_scores == 0)] = -10000000

            broadcast_alpha = alpha.view(batch_size, num_tags, 1)

            # Add all the scores together and logexp over the current_tag axis
            inner = broadcast_alpha + feats_score + transition_scores # (batch_size, num_tags, num_tags)
            alpha = (log_sum_exp(inner, 1) * mask[i].view(batch_size, 1) +
                     alpha * (1 - mask[i]).view(batch_size, 1))

        # Add end transition score
        last_tag_indexes = mask.sum(0).long() - 1
        end_transitions = self.end_transitions.expand(batch_size, num_tags) \
                            * possible_tags.transpose(0, 1).view(sequence_length * batch_size, num_tags)[last_tag_indexes + torch.arange(batch_size).to(self.device) * sequence_length]
        end_transitions[(end_transitions == 0)] = -10000000
        stops = alpha + end_transitions
        score = log_sum_exp(stops)
        return score

    def _batch2feats(self, batch):
        word_seq_tensor, char_seq_tensor, char_seq_len, label_seq_tensor = batch
        batch_size, sequence_length = word_seq_tensor.size()

        # Get embedding
        # (batch, sequence_length, word_emb_dim)
        inputs_word_emb = self.word_emb(word_seq_tensor)
        # (batch, sequence_length, word_emb_dim)
        inputs_char_emb = self.char_lstm.get_last_hiddens(char_seq_tensor, char_seq_len)

        # Combine word and chat emb
        inputs_word_char_emb = self.we_drop_layer(torch.cat([inputs_word_emb, inputs_char_emb], 2))

        tags = label_seq_tensor

        # Create mask from tags
        # PAD_INDEX is 0, other index is 1
        mask = (tags != self.PAD_INDEX)

        # Convert embedding to feature
        lstm_out, _ = self.lstm(inputs_word_char_emb)
        lstm_out = self.drop_layer(lstm_out)
        feats = self.hidden2tag(lstm_out)

        return feats, tags, mask

    def forward(self, batch):
        feats, tags, mask = self._batch2feats(batch)
        gold_score = self._score_sentence(feats, tags, mask)
        forward_score = self._forward_alg(feats, mask)
        return torch.sum(forward_score - gold_score)

    def decode(self, batch):
        feats, tags, mask = self._batch2feats(batch)
        predict_tags = self._viterbi_tags(feats, mask)
        
        return predict_tags

    def _viterbi_tags(self, feats, mask):
        """
        Parameters:
            feats: (batch_size, sequence_length, num_tags)
            mask:  Show padding tags. 0 don't calculate score. (batch_size, sequence_length)
        Returns:
            tags: (batch_size)
        """
        batch_size, sequence_length, _ = feats.shape

        feats = feats.transpose(0, 1).contiguous()
        mask = mask.transpose(0, 1).contiguous()

        # Start transition and first emission
        score = self.start_transitions + feats[0]
        history = []

        for i in range(1, sequence_length):
            broadcast_score = score.unsqueeze(2)
            broadcast_feats = feats[i].unsqueeze(1)

            next_score = broadcast_score + self.transitions + broadcast_feats
            next_score, indices = next_score.max(dim=1)

            score = torch.where(mask[i].unsqueeze(1), next_score, score)
            history.append(indices)

        # Add end transition score
        score += self.end_transitions


        # Compute the best path for each sample
        seq_ends = mask.long().sum(dim=0) - 1
        best_tags_list = []

        for idx in range(batch_size):
            _, best_last_tag = score[idx].max(dim=0)
            best_tags = [best_last_tag.item()]

            for hist in reversed(history[:seq_ends[idx]]):
                best_last_tag = hist[idx][best_tags[-1]]
                best_tags.append(best_last_tag.item())

            best_tags.reverse()
            best_tags_list.append(best_tags)

        return best_tags_list