from src.data.instance import Instance
from src.data.sentence import Sentence
from src.common.config import PAD_TAG, UNK_TAG

from tqdm import tqdm

import logging
logger = logging.getLogger(__name__)

from src.common.config import UNLABELED_TAG

class Conll2003Reader:
    def __init__(self):
        self.vocab = set()

    def load_text(self, file, to_lower = False, unlabel_to_other = False):
        instances = []
        with open(file, 'r') as f:
            words = []
            labels = []
            for line in tqdm(f.readlines()):
                line = line.rstrip()
                if line == "":
                    instances.append(Instance(Sentence(words), labels))
                    words = []
                    labels = []
                    continue
                rows = line.split(" ")
                word = rows[0]
                if to_lower:
                    word = word.lower()
                label = rows[-1]
                if unlabel_to_other and label == UNLABELED_TAG:
                    label = "O"
                words.append(word)
                labels.append(label)
                self.vocab.add(word)
        return instances