from src.common.config import UNLABELED_TAG

class Entity:
    def __init__(self, e_type, words, e_type_index):
        self.e_type = e_type
        self.words = words
        self.e_type_index = e_type_index

    def __str__(self):
        return "{}[{}]: {}".format(self.e_type, self.e_type_index, " ".join(self.words))
    
    def to_line(self):
        result = []
        if self.e_type == "O" or self.e_type == UNLABELED_TAG:
            for word in self.words:
                result.append("{} {}".format(word, self.e_type))
        else:
            if len(self.words) == 1:
                result.append("{} {}".format(self.words[0], "S-" + self.e_type))
            else:
                for idx, word in enumerate(self.words):
                    if idx == 0:
                        result.append("{} {}".format(word, "B-" + self.e_type))
                    elif idx == len(self.words) - 1:
                        result.append("{} {}".format(word, "E-" + self.e_type))
                    else:
                        result.append("{} {}".format(word, "I-" + self.e_type))
        return "\n".join(result) + "\n"
