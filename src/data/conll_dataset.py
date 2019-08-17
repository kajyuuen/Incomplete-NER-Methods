from src.data.conll_loader import Conll2003Reader
from src.common.config import PAD_TAG, UNK_TAG, UNLABELED_TAG

import torch
import numpy as np

from tqdm import tqdm

def possible_tag_masks(num_tags, tags, unlabeled_index):
    no_annotation_idx = (tags == unlabeled_index)
    tags[tags == unlabeled_index] = 0

    tags_ = torch.unsqueeze(tags, 2)
    masks = torch.zeros(tags_.size(0), tags_.size(1), num_tags)
    masks.scatter_(2, tags_, 1)
    masks[no_annotation_idx] = 1
    return masks

class Conll2003Dataset:
    def __init__(self, 
                 batch_size,
                 dataset_path,
                 word_emb_dim,
                 test_batch_size = None,
                 pretrain_type = None,
                 unlabel_to_other = False,
                 device = "cpu"):
        self.reader = Conll2003Reader()
        self.device = device
        self.batch_size = batch_size
        self.train_instances = self.reader.load_text(dataset_path + "/eng.train", unlabel_to_other=unlabel_to_other)
        self.valid_instances = self.reader.load_text(dataset_path + "/eng.testa", unlabel_to_other=unlabel_to_other)
        self.test_instances = self.reader.load_text(dataset_path + "/eng.testb", unlabel_to_other=unlabel_to_other)
        self.label2idx, self.idx2labels = self._build_label_idx()
        self.num_tags = len(self.idx2labels)
        self.word2idx, self.idx2word, self.char2idx, self.idx2char = self._build_word_idx()
        if pretrain_type == "Glove":
            self.embedding, self.embedding_dim = self._load_pretrain_embedding()
        else:
            print("No use pretrain emb")
            self.embedding, self.embedding_dim = None, word_emb_dim
        self._build_emb_table()
        self.train_batch = self.dataset2batch(self.instances2ids(self.train_instances), self.batch_size)
        self.valid_batch = self.dataset2batch(self.instances2ids(self.valid_instances), self.batch_size)
        if test_batch_size is None:
            test_batch_size = self.batch_size
        self.test_batch = self.dataset2batch(self.instances2ids(self.test_instances), test_batch_size)
        
    def _load_pretrain_embedding(self):
        embedding_dim = -1
        embedding = dict()
        with open(".vector_cache/glove.6B.100d.txt", 'r', encoding='utf-8') as file:
            for line in tqdm(file.readlines()):
                line = line.strip()
                if len(line) == 0:
                    continue
                tokens = line.split()
                if embedding_dim < 0:
                    embedding_dim = len(tokens) - 1
                else:
                    assert (embedding_dim + 1 == len(tokens))
                embedd = np.empty([1, embedding_dim])
                embedd[:] = tokens[1:]
                word = tokens[0]
                embedding[word] = embedd
        return embedding, embedding_dim

    def _build_emb_table(self):
        scale = np.sqrt(3.0 / self.embedding_dim)
        if self.embedding is not None:
            print("[Info] Use the pretrained word embedding to initialize: %d x %d" % (len(self.word2idx), self.embedding_dim))
            self.word_embedding = np.empty([len(self.word2idx), self.embedding_dim])
            for word in self.word2idx:
                if word in self.embedding:
                    self.word_embedding[self.word2idx[word], :] = self.embedding[word]
                elif word.lower() in self.embedding:
                    self.word_embedding[self.word2idx[word], :] = self.embedding[word.lower()]
                else:
                    self.word_embedding[self.word2idx[word], :] = np.random.uniform(-scale, scale, [1, self.embedding_dim])
            self.embedding = None
        else:
            self.word_embedding = np.empty([len(self.word2idx), self.embedding_dim])
            for word in self.word2idx:
                self.word_embedding[self.word2idx[word], :] = np.random.uniform(-scale, scale, [1, self.embedding_dim])

    def _build_label_idx(self):
        label2idx, idx2labels = {}, []
        label2idx[PAD_TAG] = 0
        idx2labels.append(PAD_TAG)
        for instance in self.train_instances + self.valid_instances:
            for label in instance.labels:
                if (label not in label2idx) and label != UNLABELED_TAG:
                    idx2labels.append(label)
                    label2idx[label] = len(label2idx)
        label2idx[UNLABELED_TAG] = -1
        return label2idx, idx2labels

    def _build_word_idx(self):
        word2idx, idx2word = {}, []
        word2idx[PAD_TAG] = 0
        idx2word.append(PAD_TAG)
        word2idx[UNK_TAG] = 1
        idx2word.append(UNK_TAG)

        char2idx, idx2char = {}, []
        char2idx[PAD_TAG] = 0
        idx2char.append(PAD_TAG)
        char2idx[UNK_TAG] = 1
        idx2char.append(UNK_TAG)

        for instance in self.train_instances + self.valid_instances + self.test_instances:
            for word in instance.sentence.words:
                if word not in word2idx:
                    word2idx[word] = len(word2idx)
                    idx2word.append(word)

        for instance in self.train_instances:
            for word in instance.sentence.words:
                for c in word:
                    if c not in char2idx:
                        char2idx[c] = len(idx2char)
                        idx2char.append(c)

        return word2idx, idx2word, char2idx, idx2char

    def instances2ids(self, instances):
        instances_ids = []
        for instance in instances:
            words = instance.sentence.words
            instance.word_ids = []
            instance.char_ids = []
            instance.label_ids = []
            for word in words:
                if word in self.word2idx:
                    instance.word_ids.append(self.word2idx[word])
                else:
                    instance.word_ids.append(self.word2idx[UNK_TAG])
                char_id = []
                for c in word:
                    if c in self.char2idx:
                        char_id.append(self.char2idx[c])
                    else:
                        char_id.append(self.char2idx[UNK_TAG])
                instance.char_ids.append(char_id)
            for label in instance.labels:
                if label in self.label2idx:
                    instance.label_ids.append(self.label2idx[label])
                elif label == UNLABELED_TAG:
                    instance.label_ids.append(-1)
                else:
                    raise AssertionError
            instances_ids.append(instance)
        return instances_ids

    def dataset2batch(self, instances, batch_size):
        instances_num = len(instances)
        total_batch = instances_num // batch_size + 1 if instances_num % batch_size != 0 else instances_num // batch_size
        batched_data = []
        for batch_id in range(total_batch):
            one_batch_instances = instances[batch_id * batch_size:(batch_id + 1) * batch_size]
            batched_data.append(self.instances2batch(one_batch_instances))

        return batched_data
    
    def instances2batch(self, instances):
        batch_size = len(instances)
        batch_data = sorted(instances, key=lambda instance: len(instance.sentence.words), reverse=True)
        word_seq_len = torch.LongTensor(list(map(lambda instance: len(instance.sentence.words), batch_data)))
        max_seq_len = word_seq_len.max()

        char_seq_len = torch.LongTensor([list(map(len, instance.sentence.words)) + [1] * (int(max_seq_len) - len(instance.sentence.words)) for instance in batch_data])
        max_char_seq_len = char_seq_len.max()

        word_seq_tensor = torch.zeros((batch_size, max_seq_len), dtype=torch.long)
        label_seq_tensor =  torch.zeros((batch_size, max_seq_len), dtype=torch.long)
        char_seq_tensor = torch.zeros((batch_size, max_seq_len, max_char_seq_len), dtype=torch.long)

        for idx in range(batch_size):
            word_seq_tensor[idx, :word_seq_len[idx]] = torch.LongTensor(batch_data[idx].word_ids)
            label_seq_tensor[idx, :word_seq_len[idx]] = torch.LongTensor(batch_data[idx].label_ids)

            for word_idx in range(word_seq_len[idx]):
                char_seq_tensor[idx, word_idx, :char_seq_len[idx, word_idx]] = torch.LongTensor(batch_data[idx].char_ids[word_idx])
            for word_idx in range(word_seq_len[idx], max_seq_len):
                char_seq_tensor[idx, word_idx, 0: 1] = torch.LongTensor([self.char2idx[PAD_TAG]])

        word_seq_tensor = word_seq_tensor.to(self.device)
        label_seq_tensor = label_seq_tensor.to(self.device)
        char_seq_tensor = char_seq_tensor.to(self.device)
        word_seq_len = word_seq_len.to(self.device)
        char_seq_len = char_seq_len.to(self.device)

        return [word_seq_tensor, word_seq_len, char_seq_tensor, char_seq_len, label_seq_tensor, False]
