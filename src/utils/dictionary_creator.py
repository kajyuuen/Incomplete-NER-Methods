import numpy as np

from collections import defaultdict

from src.data.conll_loader import Conll2003Reader
from src.data.entity import Entity

from src.common.config import UNLABELED_TAG

class DictionaryCreator:
    def __init__(self, file_path):
        reader = Conll2003Reader()
        self.instances = reader.load_text(file_path)
        self.e_type_counter = defaultdict(int)
        self.create_entity()

    def create_entity(self):
        entities = []
        for instance in self.instances:
            sentence_entities = []
            words = []
            for label, word in zip(instance.labels, instance.sentence.words):
                words.append(word)
                if label == "O":
                    entity = Entity(label, words, self.e_type_counter[label])
                    sentence_entities.append(entity)
                    self.e_type_counter[label] += 1
                    words = []
                else:
                    position, e_type = label.split("-")
                    if position == "S" or position == "E":
                        entity = Entity(e_type, words, self.e_type_counter[e_type])
                        sentence_entities.append(entity)
                        self.e_type_counter[e_type] += 1
                        words = []
            entities.append(sentence_entities)
        self.entities = entities
    
    def write(self, file_path):
        text = ""
        for sentence_entities in self.entities:
            for entity in sentence_entities:
                if entity.e_type == "O":
                    continue
                text += "{} {}\n".format(" ".join(entity.words), entity.e_type)

        with open(file_path, mode="w") as f:
            f.write(text)
