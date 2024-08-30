**@2024, Qingan Yan (reference or citation is required for re-posting)**

## Preface
A personal reading log about the light-weight network paper: 
[Howard et al, MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications](https://arxiv.org/abs/1704.04861). 

The note serves mainly for my understanding enhancement but open to any comments and discussions. I would like to list several keywords here, facilitating my potential memory querying and revisiting in the future.

{mobilenet}, {separable convolution}, {1*1 convolution}

## Motivation
Standard 2D convolution operation can be factorized into one convolution processing as per each spatial map and one fusing along depth channel subsequently. This linear combination can substantially reduce network computation.

## Depthwise Separable Convolution
Normally a 2D convolution kernel shapes like $[D_I, H_K, W_K]$ which jointly considers values in spatial domain as well as along depth channel. So for a input map $[D_I, H, W]$, a standard 2D convolution layer produces a new $[D_O, H, W]$ feature map if no downsampling is applied, in which there are $D_O$ 2D convolution kernels $[D_I, H_K, W_K]$ operating on each _pixel_, i.e., a computation equals to $D_O \times (D_I \times H_K \times W_K) \times (H \times W)$. Note that for the sake of distinction in the following description, I refer the _element_ to be more primitive scalar, while _pixel_ stands for vector format with depth dimension.

The depthwise separable convolution breaks the joint spatial and depth consideration into two sequential steps: (1) A depthwise $[1, H_K, W_K]$ convolution applied to each element, i.e., $H_K \times W_K \times D_I \times H \times W$ cost; (2) $D_O$ size of pointwise $1 \times 1$ convolutions $[D_I, 1, 1]$ performed on each pixel, i.e., $D_O \times D_I \times H \times W$. The functionality of $1 \times 1$ convolution in an efficient way fuses information across channels and generates a series of new maps. Therefore, the computation reduces to $H_K \times W_K \times D_I \times H \times W + D_O \times D_I \times H \times W$.

The structure of depthwise separable convolution is then expressed as:

$3 \times 3$ Depthwise Conv $\longrightarrow$ BN $\longrightarrow$ ReLU $\longrightarrow$ $1 \times 1$ Conv $\longrightarrow$ BN $\longrightarrow$ ReLU,
 
which is approximating standard convolution:

$3 \times 3$ Conv $\longrightarrow$ BN $\longrightarrow$ ReLU.

## Summary
The idea is quite simple and effective. Similar to the separable filter in conventional image process, the algorithm also splits a convolution operator into two steps, but in depth&point-wise instead of horizontal&vertical-wise.
