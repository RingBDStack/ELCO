# Graph Convolutional Networks and Graph Attention Networks

The original codes of GCN and GAT are:
GCN: https://github.com/tkipf/gcn
GAT: https://github.com/Diego999/pyGAT
The modified codes are mainly located in the function ‘load_data’ in utils.py.

Please note that, compared with the submitted paper, we have acheived even *higher* scores in the last 10 independent trials using ELCO-GCN.
- Cora: 87, 86.7, 86.9, 87, 86.7, 87, 86.1, 87, 87.3, 86.8 = 86.85 0.30083
- Citeseer: 77.8, 76.5, 76.9, 76.8, 76.6, 77.1, 77.6, 76.2, 76.5, 77.9 = 76.99 0.5629
- Pubmed: 84.1, 84.3, 84.4, 84.4, 84.7, 84.3, 83.9, 84.3, 84.6, 84.3 = 84.33 0.2147

## Installation

```bash
python setup.py install
```

## Core Requirements
* tensorflow (>0.12)
* networkx
* scikit-learn
* karateclub

Please refer to requirements.txt for the full list.

## Run our demo

```bash
cd gcn
python train.py
```

## GCN-ELCO Data

We have already prepared citation network data (Cora, Citeseer or Pubmed). 

If you prefer to download the original data again, please run download_data.sh (for linux only)

You can specify the tested dataset as follows:

```bash
python train.py --dataset citeseer
```

(or by editing `train.py`)

If you want to skip the pre-process, you can directrly load the intermediate results after the first trial (do not skip by default):

change the 'skip = 0' to 'skip = 1' in function 'load_data' in utils.py.

## Tested Environment

OS: Microsoft Windows 10 Professional Edition (64-bit)
CPU: Intel(R) Core(TM) i7-9750H CPU @ 2.60GHz(2592 MHz)
Mem: 16GB
