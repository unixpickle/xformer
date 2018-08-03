# xformer

This is a lightweight implementation of the decoder from the [Transformer architecture](https://arxiv.org/abs/1706.03762). It is implemented both as a direct function on `[batch x timesteps x N]` sequences, and as an RNNCell.