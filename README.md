# Fuzzy-LSTM-CRF in PyTorch

Implementation of Fuzzy-LSTM-CRF and LSTM-CRF for sequence labeling.

## Usage

### Input data format

```
token label
token label
token label
```

Unlabeled token's tag is `NOANNOTATION`.

### Delete label

```
$ python src/labeling/delete_annotation.py conll2003_bioes/eng.train tmp.train --entity_keep_ratio 0.4
```

## Create Dictionary

```
$ python src/labeling/dict_create.py datasets/dict_conll2003_bioes/eng.testa datasets/dict_conll2003_bioes/testa.dic
```

## Annotation using Dictionary

```
$ python src/labeling/dict_labeling.py datasets/dict_conll2003_bioes/testa.dic datasets/dict_conll2003_bioes/eng.train
```

## Train and Predict

### BiLSTM-CRF

```
$ python main.py config/lample.json
```

### BiLSTM-Fuzzy-CRF


```
$ python main.py config/fuzzy_lample.json
```

### Hard Approach

```
$ python main.py config/hard.json
```

## References

- https://github.com/kmkurn/pytorch-crf
- https://github.com/threelittlemonkeys/lstm-crf-pytorch
- https://github.com/kolloldas/torchnlp/blob/master/torchnlp
- https://github.com/allanj/pytorch_lstmcrf
- https://github.com/shangjingbo1226/AutoNER

