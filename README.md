## Adjacent Attention Network (wip)

An implementation of a simple transformer that is equivalent to graph neural network where the message passing is done with multi-head attention at each successive layer. Since Graph Attention Network is already taken, I decided to name it Adjacent Attention Network instead. The design will be more transformer-centric. Instead of using the square root inverse adjacency matrix trick by Kipf and Welling, in this framework it will simply be translated to the proper attention mask at each layer.

This repository is for my own exploration into the graph neural network field. My gut tells me the transformers architecture can generalize and outperform graph neural networks.

## Install

```bash
$ pip install adjacent-attention-network
```

## Usage

```python
import torch
from adjacent_attention_network import AdjacentAttentionNetwork

model = AdjacentAttentionNetwork(
    dim = 512,
    depth = 6,
    heads = 4
)

adj_mat = torch.empty(1, 1024, 1024).uniform_(0, 1) < 0.1
nodes   = torch.randn(1, 1024, 512)

model(nodes, adj_mat) # (1, 1024, 512)
```
