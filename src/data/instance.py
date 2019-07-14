class Instance:
    def __init__(self, sentence, labels):
        self.sentence = sentence
        self.labels = labels
        self.word_ids = None
        self.char_ids = None
        self.label_ids = None

    def __len__(self):
        return len(self.sentence)
