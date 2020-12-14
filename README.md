## Adjacent Attention Network

An implementation of a simple transformer that is equivalent to graph neural network where the message passing is done with multi-head attention at each successive layer. Since Graph Attention Network is already taken, I decided to name it Adjacent Attention Network instead. The design will be more transformer-centric. Instead of using the square root inverse adjacency matrix trick by Kipf and Welling, in this framework it will simply be translated to the proper attention mask at each layer.

This repository is for my own exploration into the graph neural network field. My gut tells me the transformers architecture can generalize and outperform graph neural networks.

## Install

```bash
$ pip install adjacent-attention-network
```

## Usage

Basically a transformers where each node pays attention to the neighbors as defined by the adjacency matrix. Complexity is O(n * max_neighbors). Max number of neighbors as defined by the adjacency matrix.

The following example will have a complexity of ~ 1024 * 100

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
mask    = torch.ones(1, 1024).bool()

model(nodes, adj_mat, mask = mask) # (1, 1024, 512)
```

If the number of neighbors contain outliers, then the above will lead to wasteful computation, since a lot of nodes will be doing attention on padding. You can use the following stop-gap measure to account for these outliers.

```python
import torch
from adjacent_attention_network import AdjacentAttentionNetwork

model = AdjacentAttentionNetwork(
    dim = 512,
    depth = 6,
    heads = 4,
    num_neighbors_cutoff = 100
).cuda()

adj_mat = torch.empty(1, 1024, 1024).uniform_(0, 1).cuda() < 0.1
nodes   = torch.randn(1, 1024, 512).cuda()
mask    = torch.ones(1, 1024).bool().cuda()

# for some reason, one of the nodes is fully connected to all others
adj_mat[:, 0] = 1.

model(nodes, adj_mat, mask = mask) # (1, 1024, 512)
```

For non-local attention, I've decided to use a trick from the Set Transformers paper, the Induced Set Attention Block (ISAB). From the lens of graph neural net literature, this would be analogous as having global nodes for message passing non-locally.

```python
import torch
from adjacent_attention_network import AdjacentAttentionNetwork

model = AdjacentAttentionNetwork(
    dim = 512,
    depth = 6,
    heads = 4,
    num_neighbors_cutoff = 100,
    num_global_nodes = 5
).cuda()

adj_mat = torch.empty(1, 1024, 1024).uniform_(0, 1).cuda() < 0.1
nodes   = torch.randn(1, 1024, 512).cuda()
mask    = torch.ones(1, 1024).bool().cuda()

# for some reason, one of the nodes is fully connected to all others
# perhaps some instagram influencer
adj_mat[:, 0] = 1.

model(nodes, adj_mat, mask = mask) # (1, 1024, 512)
```
