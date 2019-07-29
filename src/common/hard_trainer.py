import os

import torch
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence

from tqdm import tqdm
from seqeval.metrics import classification_report
from seqeval.metrics import f1_score
from sklearn.model_selection import train_test_split

from src.common.convert import convert

import logging
logger = logging.getLogger(__name__)

class HardTrainer():
    def __init__(self,
                 model,
                 train_iter,
                 valid_iter = None,
                 test_iter = None,
                 label_dict = None,
                 learning_rate = 0.01,
                 weight_decay = 1e-8,
                 clipping = None,
                 save_path = None,
                 force_save = False,
                 train_only = False,
                 early_stopping = None,
                 device = "CPU"
                 ):
    
        self.model = model
        self.device = device
        self.split_train_iters = train_test_split(train_iter, test_size=0.5, random_state=42)
        self.valid_iter = valid_iter
        self.test_iter = test_iter
        self.label_dict = label_dict
        self.test_flag = self.test_iter is not None and self.label_dict is not None
        self.optimizer = optim.SGD(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.save_path = save_path
        self.clipping = clipping
        self.train_only = train_only

        if self.save_path:
            os.makedirs(self.save_path, exist_ok = True)

        if self.test_flag and os.path.exists(self.save_path + "/result.txt"):
            os.remove(self.save_path + "/result.txt")

    def _init_train(self, dataset_iter, num_epochs):
        self._simple_train(dataset_iter, num_epochs, "Init Train", init_train=True)
    
    def _annotation_by_restricted_viterbi(self, dataset_iter):
        for i in range(len(dataset_iter)):
            predict_tags = self.model.restricted_forward(dataset_iter[i])
            dataset_iter[i][3] = torch.LongTensor(predict_tags).to(self.device)
        return dataset_iter

    def _simple_train(self, dataset_iter, num_epochs, description, early_stopping=None, init_train=False):
        for ind in range(num_epochs):
            self.model.train()
            epoch_loss = 0
            count = 0

            progress_iter = tqdm(dataset_iter, leave=False)

            for batch in progress_iter:
                self.optimizer.zero_grad()
                loss = self.model.neg_log_likelihood(batch, init_train=init_train)
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()
                count += 1

                if self.clipping is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clipping) 

                loss = epoch_loss/count

                progress_iter.set_description('{}'.format(description))
                progress_iter.set_postfix(loss=(loss))
        return loss

    def train(self, num_epochs, sub_num_epochs=10, early_stopping=None, save=True):

        best_iteration = 0
        best_loss = 10e7

        self._init_train(self.split_train_iters[0], sub_num_epochs)

        for epoch_ind in range(num_epochs):
            for i in range(2)[::-1]:
                labeled_train_iter = self._annotation_by_restricted_viterbi(self.split_train_iters[i])
                train_loss = self._simple_train(labeled_train_iter, sub_num_epochs, "Train")

            if self.save_path is not None:
                path = self.save_path + "/epoch_{}".format(epoch_ind+1)
                torch.save(self.model.state_dict(), path)
                with open(self.save_path + "/best_epoch.txt", "w") as f:
                    f.write(str(best_iteration))

            if self.train_only:
                continue

            # Valid
            self.model.eval()
            if self.valid_iter is not None:

                epoch_valid_loss = 0
                valid_count = 0

                progress_iter = tqdm(self.valid_iter, leave=False)

                for batch in progress_iter:
                    loss = self.model.neg_log_likelihood(batch)

                    epoch_valid_loss += loss.item()
                    valid_count += 1
                    valid_loss = epoch_valid_loss/valid_count

                    progress_iter.set_description('Validing')
                    progress_iter.set_postfix(valid_loss=(valid_loss))

            if self.valid_iter is None:
                loss_text = "Train Loss: {}".format(train_loss)
                if train_loss < best_loss:
                    best_loss = train_loss
                    best_iteration = epoch_ind+1
            else:
                loss_text = "Train Loss: {}, Valid Loss: {}".format(train_loss, valid_loss)
                if valid_loss < best_loss:
                    best_loss = valid_loss
                    best_iteration = epoch_ind+1
            logger.info(loss_text)

            # Test
            self.model.eval()
            if self.test_flag:
                y_true, y_pred = [], []
                for batch in self.test_iter:
                    predict_tags = self.model(batch)
                    _, _, _, label_seq_tensor = batch
                    y_true.extend(convert(label_seq_tensor.tolist(), self.label_dict))
                    y_pred.extend(convert(predict_tags, self.label_dict))
                with open(self.save_path + "/result.txt", "a") as f:
                    f.write("\nepoch: {}, {}, F1: {}\n".format(epoch_ind+1, loss_text, f1_score(y_true, y_pred)))
                    f.write(classification_report(y_true, y_pred, digits=5))

            # Early stopping
            if early_stopping is not None:
                if best_iteration - epoch_ind+1  == early_stopping:
                    logger.info("Best Epoch: {}".format(best_iteration))

