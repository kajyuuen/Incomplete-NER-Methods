# BiLSTM-FuzzyCRF in PyTorch

Implementation of BiLSTM-Fuzzy-CRF and BiLSTM-CRF for sequence labeling.

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
$ python delete_annotation.py conll2003_bioes/eng.train tmp.train --entity_keep_ratio 0.4
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

## References

- https://github.com/kmkurn/pytorch-crf
- https://github.com/threelittlemonkeys/lstm-crf-pytorch
- https://github.com/kolloldas/torchnlp/blob/master/torchnlp
- https://github.com/allanj/pytorch_lstmcrf