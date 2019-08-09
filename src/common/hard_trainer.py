import os
import copy

import torch
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence

from tqdm import tqdm
from seqeval.metrics import classification_report
from seqeval.metrics import f1_score
from seqeval.metrics import accuracy_score
from seqeval.metrics import recall_score
from sklearn.model_selection import train_test_split

from src.common.convert import convert

import logging
logger = logging.getLogger(__name__)

class HardTrainer():
    def __init__(self,
                 model,
                 train_iter,
                 valid_iter,
                 test_iter,
                 label_dict = None,
                 sub_num_epochs = 2,
                 learning_rate = 0.05,
                 weight_decay = 0,
                 clipping = None,
                 save_path = None,
                 force_save = False,
                 train_only = False,
                 early_stopping = None,
                 eval_freq = 2000,
                 device = "CPU"
                 ):
    
        self.init_model = model
        self.device = device
        self.train_iter = train_iter
        self.sub_num_epochs = sub_num_epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.split_train_iters = train_test_split(train_iter, test_size=0.5, random_state=42)
        self.valid_iter = valid_iter
        self.test_iter = test_iter
        self.label_dict = label_dict
        self.test_flag = self.test_iter is not None and self.label_dict is not None
        self.save_path = save_path
        self.init_model_path = save_path + "/init_model"
        self.clipping = clipping
        self.eval_freq = eval_freq
        self.train_only = train_only

        if self.save_path:
            os.makedirs(self.save_path, exist_ok = True)

        if self.test_flag and os.path.exists(self.save_path + "/result.txt"):
            os.remove(self.save_path + "/result.txt")
    
    def _annotation_by_restricted_viterbi(self, model, dataset_iter):
        for i in range(len(dataset_iter)):
            predict_tags = model.restricted_forward(dataset_iter[i])
            dataset_iter[i][3] = torch.LongTensor(predict_tags).to(self.device)
        return dataset_iter

    def _simple_score(self, model, iter):
        model.eval()
        y_true, y_pred = [], []
        for batch in iter:
            predict_tags = model(batch)
            _, _, _, label_seq_tensor = batch
            y_true.extend(convert(label_seq_tensor.tolist(), self.label_dict))
            y_pred.extend(convert(predict_tags, self.label_dict))
        return accuracy_score(y_true, y_pred), recall_score(y_true, y_pred), f1_score(y_true, y_pred)

    def _simple_valid(self, model, valid_iter):
        model.eval()

        epoch_valid_loss = 0
        valid_count = 0

        progress_iter = tqdm(valid_iter, leave=True)
        for batch in progress_iter:
            loss = model.neg_log_likelihood(batch)

            epoch_valid_loss += loss.item()
            valid_count += 1
            valid_loss = epoch_valid_loss/valid_count

            progress_iter.set_description('Validing')
            progress_iter.set_postfix(valid_loss=(valid_loss))

        return valid_loss

    def _simple_train(self, model, dataset_iter, valid_iter, num_epochs, description, save_path, early_stopping=None, init_train=False):
        best_iteration, best_f1 = 0, 0
        best_valid_score = [0] * 3
        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

        for ind in range(num_epochs):
            epoch_loss = 0
            count = 0

            progress_iter = tqdm(dataset_iter, leave=True)
            model.train()
            for batch in progress_iter:
                optimizer.zero_grad()
                loss = model.neg_log_likelihood(batch, init_train=init_train)
                loss.backward()

                if self.clipping is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.clipping)

                optimizer.step()

                epoch_loss += loss.item()
                count += 1

                loss = epoch_loss/count

                progress_iter.set_description('{}'.format(description))
                progress_iter.set_postfix(loss=(loss))

                if count != 0 and (count % self.eval_freq == 0 or count == len(progress_iter)):
                    os.makedirs(save_path, exist_ok = True)

                    # Valid
                    valid_precision, valid_recall, valid_f1 = self._simple_score(model, valid_iter)

                    with open(save_path + "/valid_score.txt", "a") as f:
                        f.write("[{}] Precision: {}, Recall: {}, F1: {}\n".format(ind+1, valid_precision, valid_recall, valid_f1))

                    if valid_f1 >= best_f1:
                        best_f1 = valid_f1
                        best_iteration = ind+1
                        best_valid_score = valid_precision, valid_recall, valid_f1

                        # Save
                        path = save_path + "/best_model"
                        torch.save(model.state_dict(), path)

                    model.train()

            # Early stopping
            if early_stopping is None:
                continue
            if best_iteration - (ind+1)  == early_stopping:
                logger.info("Best Epoch: {}".format(best_iteration))
                break

        return best_valid_score

    def _load_model(self, path):
        model = copy.deepcopy(self.init_model)
        model.load_state_dict(torch.load(path))
        return model

    def _load_best_model(self, k, iteration):
        path = self.save_path + "/k_{}/iteration_{}".format(k, iteration)
        best_model_path = path + "/best_model"
        return self._load_model(best_model_path)


    def train(self, num_epochs, early_stopping=None, save=True):
        # Init train
        for i in range(2)[::-1]:
            model = copy.deepcopy(self.init_model)
            self._simple_train(model, self.split_train_iters[i], self.valid_iter, self.sub_num_epochs, "Init Train", self.save_path + "/k_{}/iteration_{}".format(i, 0), init_train=True)

        for epoch_ind in range(num_epochs):
            for k in range(2)[::-1]:
                # Annotation
                if k == 0:
                    other_k = 1
                else:
                    other_k = 0
                model = self._load_best_model(other_k, epoch_ind)
                labeled_train_iter = self._annotation_by_restricted_viterbi(model, self.split_train_iters[k])

                # Train
                model = copy.deepcopy(self.init_model)
                self._simple_train(model, labeled_train_iter, self.valid_iter, self.sub_num_epochs, "Train", self.save_path + "/k_{}/iteration_{}".format(k, epoch_ind+1), early_stopping=5)

            if True:
                # Final Train
                model = copy.deepcopy(self.init_model)
                best_valid_score = self._simple_train(model, self.train_iter, self.valid_iter, self.sub_num_epochs, "Final Train", self.save_path + "/final", early_stopping=5)
                loss_text = "[Best Valid] Precision: {}, Recall: {}, F1:{}".format(best_valid_score[0], best_valid_score[1], best_valid_score[2])

                # Test
                model.eval()
                if self.test_flag:
                    y_true, y_pred = [], []
                    for batch in self.test_iter:
                        predict_tags = model(batch)
                        _, _, _, label_seq_tensor = batch
                        y_true.extend(convert(label_seq_tensor.tolist(), self.label_dict))
                        y_pred.extend(convert(predict_tags, self.label_dict))
                    with open(self.save_path + "/result.txt", "a") as f:
                        f.write("\nepoch: {}, {}, F1: {}\n".format(epoch_ind+1, loss_text, f1_score(y_true, y_pred)))
                        f.write(classification_report(y_true, y_pred, digits=5))

        # Final Train
        model = copy.deepcopy(self.init_model)
        best_valid_score = self._simple_train(model, self.train_iter, self.valid_iter, self.sub_num_epochs, "Final Train", self.save_path + "/final", early_stopping=5)
        loss_text = "[Best Valid] Precision: {}, Recall: {}, F1:{}".format(best_valid_score[0], best_valid_score[1], best_valid_score[2])

        # Test
        model.eval()
        if self.test_flag:
            y_true, y_pred = [], []
            for batch in self.test_iter:
                predict_tags = model(batch)
                _, _, _, label_seq_tensor = batch
                y_true.extend(convert(label_seq_tensor.tolist(), self.label_dict))
                y_pred.extend(convert(predict_tags, self.label_dict))
            with open(self.save_path + "/result.txt", "a") as f:
                f.write("\nepoch: {}, {}, F1: {}\n".format(epoch_ind+1, loss_text, f1_score(y_true, y_pred)))
                f.write(classification_report(y_true, y_pred, digits=5))