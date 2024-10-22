**@ 2024 Qingan Yan. Reference or citation is required for re-posting.**

## Preface
A personal reading log about the end-to-end 3D reconstruction paper: 
[S. Wang, et al. DUSt3R: Geometric 3D Vision Made Easy, CVPR 2024](https://arxiv.org/abs/2312.14132). 

The note serves mainly for my understanding enhancement but is open to any comments and discussions. I would like to list several keywords here, facilitating my potential memory querying and revisiting in the future.

{all-in-one}, {vit}, {siamese}, {confidence}, {global alignment}

## Motivation
Classic 3D reconstruction pipelines involve a sequence of sub-problems, like feature matching, sparse SfM, bundle adjustment and dense MVS. Therefore it would be appreciate to have an end-to-end 3D reconstruction pipeline that regresses point clouds directly from one or multiple un-calibrated and un-posed images.

## Method
__Intuition__. In order to design an end-to-end system, a straightforward idea is to train a network learning the mapping from 2D pixels to 3D points with groundtruth 3D points supervision. The paper also designed like this. Therefore, the remaining challenge is how to apply losses. If supervised via reprojection errors, like bundle adjustment, camera poses and pixel correspondences either are pre-required or have to be jointly trained. Instead, the work simply leverage a regression loss, directly comparing 3D Euclidean distance estimated for each pixel, which resembles calculating depth. That means, rather than explicitly establishing hard but fragile pixelwise corelation which might be suffered by blurry and textureless signals, the algorithm encourages itself to implicitly learn associations that would most benefit the task, upon more holistic and highlevel cross-attention understanding. Furthermore, naive per view estimation loses important geometric co-regulations. So the algorithm acts in a stereo mode. That is, each time two views are inputted and the network end-to-end outputs two point clouds in the same coordinate. A series of downstream applications can be therefore performed on the output _pointmap_ representation.

__Pointmap__. Something quite likes depth map. It is a map with the same resolution to input image but the data are $xyz$-coordinate values $X \in \text{R}^{W * H * 3}$. It can also be regarded as a point cloud that each point corresponds to the pixel, i.e., $I_{i, j} \leftrightarrow X_{i, j}$.

__Unprojection__. Given a groundtruth depth map $D \in \text{R}^{W * H}$ and intrinsic matrix $K$, the 3D point in $n$ th-camera coordinate should be
```math
X^{n}_{i, j} = K^{-1} [i \times D_{i, j}, j \times D_{i, j}, 1 \times D_{i, j}]^T. \quad (1)
```
The point in $m$ th-camera coordinate is 
```math
X^{m, n} = P_m P^{-1}_n h(X^{n}), \quad (2)
```
with $P \in \text{R}^{3 \times 4}$ the world-to-camera pose and the homogeneous mapping $h:(x, y, z) \rightarrow (x, y, z, 1)$. Eq.(1) and (2) are used mainly to generate groundtruth 3D point supervisions from depth maps later. 

__Input and Output__. Input two RGB images $I^1, I^2 \in \text{R}^{W * H * 3}$, output two pointmaps $X^{1, 1}, X^{2, 1} \in \text{R}^{W * H * 3}$ with associated confidence maps $C^{1, 1}, C^{2, 1} \in \text{R}^{W * H}$ both in the coordinate of $I^1$.

__Network Architecture__. Since two images serve as input, so Siamese structure is adopted where two identical branches with weight-sharing encode each image separately, i.e., $F^1 = Encoder(I^1)$, $F^2 = Encoder(I^2)$. The encoder specifically is ViT-Large. The decoder follows ViT-Base, comprising $B$ repeating functional blocks and a MLP head, and initially $G^1_0 = F^1$ and $G^2_0 = F^2$. Each block sequentially performs self-attention and cross-attention. Self-attention draws relationship within one image, yet cross-attention mixes knowledge with the other image.
```math
G^1_i = \text{DecoderBlock}^1_i(G^1_{i - 1}, G^2_{i - 1}),
```
```math
G^2_i = \text{DecoderBlock}^2_i(G^2_{i - 1}, G^1_{i - 1}),
```
```math
X^{1, 1}, C^{1, 1} = \text{DPTHead}^1(G^1_{0}, G^1_{1}, \dots, G^1_{B}),
```
```math
X^{2, 1}, C^{2, 1} = \text{DPTHead}^2(G^2_{0}, G^2_{1}, \dots, G^2_{B}).
```

__Training Objective__. The loss could be quite straightforward. For each point in the estimated pointmap, we compare it with the groundtruth pointmap $\bar{X}$ which is derived from depth map via Eq.(1) for $I^1$ and Eq.(2) for $I^2$. So for each valid pixel $i$ with groundtruth depth $i \in D^v$ and $v \in \{1, 2\}$, the loss is expressed:
```math
l_{regr}(v, i) = \| \frac{1}{z} X^{v, 1}_i - \frac{1}{\bar{z}} \bar{X}^{v, 1}_i \|. \quad (3)
```
To mitigate the scale ambiguity, $z$ serves as a normalization factor which expresses the average distance of all valid points to the origin:
```math
z = \frac{1}{\sum \limits_{v \in \{1, 2\}} |D^v|} \sum \limits_{v \in \{1, 2\}} \sum \limits_{i \in D^v} \| X^v_i \|. \quad (4)
```
P.s., I don't check the code for if $\| X^v_i \|$ should be square-rooted here or not.

__Confidence Map__. As some estimates are ill-defined, e.g., in the sky or on translucent objects, it is thus essential to have a coefficient downgrade uncertain parts. That's what the estimated confidence maps are exactly designed for.
```math
L_{conf} = \sum \limits_{v \in \{1, 2\}} \sum \limits_{i \in D^v} C^{v, 1}_i l_{regr}(v, i) - \alpha \ln C^{v, 1}_i, \quad (5)
```
where $\alpha \ln C^{v, 1}_i$ is a regularization term to refrain from $C^{v, 1}_i \approx 0$ as it will receive a large penalty $\ln 0 \rightarrow -\infty$. So the authors define $\hat{C}^{v, 1}_i = 1 + \exp C^{v, 1}_i > 1$.