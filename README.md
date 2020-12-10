## Adjacent Attention Network (wip)

An implementation of a simple transformer that is equivalent to graph neural network where the message passing is done with multi-head attention at each successive layer. Since Graph Attention Network is already taken, I decided to name it Adjacent Attention Network instead. The design will be more transformer-centric. Instead of using the square root inverse adjacency matrix trick by Kipf and Welling, in this framework it will simply be translated to the proper attention mask at each layer.

This repository is for my own exploration into the graph neural network field. My gut tells me the transformers architecture can generalize and outperform graph neural networks.
