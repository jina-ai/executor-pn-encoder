# PNEncoder

**PNEncoder** is an Executor that receives Documents containing point sets data in its blob attribute, with shape 
(2048, 3) and encodes it to embeddings of shape (128,). **PNEncoder** uses the Pointnet model to create embeddings.

This Executor offers a GPU tag to speed up encoding. For more information on how to run the executor on GPU, check out 
[the documentation](https://docs.jina.ai/tutorials/gpu-executor/).


## Reference

- [Pointnet paper](https://arxiv.org/abs/1612.00593)
- [Pointnet GitHub Repository](https://github.com/charlesq34/pointnet)

