# Graph Attention Network

This is a pytorch implementation of the Graph Attention Network (GAT)

The GAT code is the original code, and its input is the output of ELCO-GCN (i.e., the data already augmented by ELCO)

## Run our demo

```bash
python train.py
```

## Data

We have already prepared the data. You may also copy *.cites, *.content, and *.extra from ..\ELCO\ELCO-GCN(run first)\gcn for a whole-process reproduction.

You can specify a dataset as follows:

- change 'def load_data(path="./data/citeseer/", dataset="citeseer"):' in utils.py into corresponding dataset (cora, citeseer, pubmed)
- change 'idx_train = list(range(120)) + extra_train' in utils.py into corresponding dataset (cora: 140, citeseer: 120, pubmed: 60)

## Requirements

pyGAT requires Python 3.5 and PyTorch 0.4.1.

