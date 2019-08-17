import os

import torch
import torch.optim as optim

from tqdm import tqdm
from seqeval.metrics import classification_report
from seqeval.metrics import f1_score

from src.common.convert import convert

import logging
logger = logging.getLogger(__name__)

class Trainer():
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
                 early_stopping = None
                 ):
    
        self.model = model
        self.train_iter = train_iter
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

    def train(self, num_epochs, early_stopping=None, save=True):

        best_iteration = 0
        best_loss = 10e7

        for ind in range(num_epochs):
            # Train
            self.model.train()
            epoch_train_loss = 0
            train_count = 0

            progress_iter = tqdm(self.train_iter, leave=False)

            for batch in progress_iter:
                self.optimizer.zero_grad()
                loss = self.model(batch)
                loss.backward()

                if self.clipping is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clipping) 

                self.optimizer.step()

                epoch_train_loss += loss.item()
                train_count += 1

                train_loss = epoch_train_loss/train_count

                progress_iter.set_description('Training')
                progress_iter.set_postfix(train_loss=(train_loss))
            
            if self.save_path is not None:
                path = self.save_path + "/epoch_{}".format(ind+1)
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
                    loss = self.model(batch)

                    epoch_valid_loss += loss.item()
                    valid_count += 1
                    valid_loss = epoch_valid_loss/valid_count

                    progress_iter.set_description('Validing')
                    progress_iter.set_postfix(valid_loss=(valid_loss))

            if self.valid_iter is None:
                loss_text = "Train Loss: {}".format(train_loss)
                if train_loss < best_loss:
                    best_loss = train_loss
                    best_iteration = ind+1
            else:
                loss_text = "Train Loss: {}, Valid Loss: {}".format(train_loss, valid_loss)
                if valid_loss < best_loss:
                    best_loss = valid_loss
                    best_iteration = ind+1
            logger.info(loss_text)

            # Test
            self.model.eval()
            if self.test_flag:
                y_true, y_pred = [], []
                for batch in self.test_iter:
                    predict_tags = self.model.decode(batch)
                    _, _, _, _, label_seq_tensor = batch
                    y_true.extend(convert(label_seq_tensor.tolist(), self.label_dict))
                    y_pred.extend(convert(predict_tags, self.label_dict))
                with open(self.save_path + "/result.txt", "a") as f:
                    f.write("\nepoch: {}, {}, F1: {}\n".format(ind+1, loss_text, f1_score(y_true, y_pred)))
                    f.write(classification_report(y_true, y_pred, digits=5))

            # Early stopping
            if early_stopping is not None:
                if best_iteration - ind+1  == early_stopping:
                    logger.info("Best Epoch: {}".format(best_iteration))
