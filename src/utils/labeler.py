import numpy as np

from collections import defaultdict

from tqdm import tqdm

from src.data.conll_loader import Conll2003Reader
from src.data.entity import Entity

from src.common.config import UNLABELED_TAG

class Labeler:
    def __init__(self, dict_file_path, target_file, to_lower=False):
        np.random.seed(42)
        reader = Conll2003Reader()
        self.to_lower = to_lower
        self.e_type_counter = defaultdict(int)
        self.entity_dict = self.load_dict(dict_file_path)
        self.instances = reader.load_text(target_file, to_lower)
        self._delete_label()

    def load_dict(self, dict_file_path):
        entity_dict = defaultdict(list)
        with open(dict_file_path) as f:
            lines = [ line.strip() for line in f.readlines() ]

        for line in lines:
            line = line.split(" ")
            e_type, words = line[-1], line[:-1]
            if self.to_lower:
                words = [ word.lower() for word in words ]
            entity = Entity(e_type, words, self.e_type_counter[e_type])
            entity_dict[len(words)].append(entity)
            self.e_type_counter[e_type] += 1

        entity_dict = sorted(entity_dict.items(), key=lambda x:x[0])
        return entity_dict

    def _entity_type2labels(self, entity_type, length):
        if entity_type == "O":
            raise AssertionError
        if length == 1:
            return ["S-" + entity_type]
        labels = []
        for i in range(length):
            if i == 0:
                labels.append("B-" + entity_type)
            elif i == length - 1:
                labels.append("E-" + entity_type)
            else:
                labels.append("I-" + entity_type)
        return labels

    def _delete_label(self):
        for instance_idx in range(len(self.instances)):
            self.instances[instance_idx].labels = [UNLABELED_TAG] * len(self.instances[instance_idx].labels)

    def match(self):
        pbar = tqdm(total=sum([ len(entities) for _, entities in self.entity_dict]))
        for length, entities in self.entity_dict:
            for entity in entities:
                for instance_idx in range(len(self.instances)):
                    for i in range(len(self.instances[instance_idx].sentence.words)-length+1):
                        if self.instances[instance_idx].sentence.words[i:i+length] == entity.words:
                            self.instances[instance_idx].labels[i:i+length] = self._entity_type2labels(entity.e_type, length)
                pbar.update(1)
        pbar.close()

    def write(self, file_path):
        text = ""
        for instance in self.instances:
            for word, label in zip(instance.sentence.words, instance.labels):
                text += "{} {}\n".format(word, label)
            text += "\n"

        with open(file_path, mode="w") as f:
            f.write(text)
