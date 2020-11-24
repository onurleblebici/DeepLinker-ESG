DeepLinker-ESG -- An Implementation of DeepLinker for Event Sequence Graph Link Prediction Tasks
===============================================================================

About
-----

DeepLinker uses fixed neighborhoods on mini batch sampling strategy. DeepLinker shares similar architecture with GraphSAGE. Differences between them are in sampling strategy and using GAT instead of GCN. The proposed DeepLinker model is used to create a hidden representation of each node, using the attention mechanism that is shared by the node's neighbors. 

Original DeepLinker is extended in some aspects as follows.
-----

There was and no parametric data input support to work with other training data. Along with this, a feature that can load the outputs of the mxe_parser application has been added. Only gpu support was available, cpu support is added. Test evaluation metrics are calculated at the end of the each epoch. Training, validation and test results were printed on the screen by the application, all the results are written in a csv formatted file at the end of each iteration.

Original DeepLinker implementation
-----

The original implementation of DeepLinker is [here](https://github.com/Villafly/DeepLinker).

The original paper is:
> Weiwei Gu, Fei Gao, Xiaodan Lou, and Jiang Zhang, Link Prediction via Graph Attention Network. [\[PDF\]](https://arxiv.org/pdf/1910.04807.pdf)

Reference
---------

If you find the code useful, please cite our paper.

    @inproceedings{
      title={Application of Graph Neural Networks on Software Modeling},
      author={Onur Yusuf Leblebici},
      year={2020}
    }