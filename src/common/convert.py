from src.common.config import PAD_TAG

def convert(predict_tags, label2idx):
    index_dict = {v: k for k, v in label2idx.items()}
    PAD_INDEX = label2idx[PAD_TAG]
    labels = []
    for predict_tag in predict_tags:
        label = [ index_dict[p] for p in predict_tag if p != PAD_INDEX ]
        labels.append(label)
    return labels