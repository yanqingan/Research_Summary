**@ 2024 Qingan Yan. Reference or citation is required for re-posting.**

## Preface
A personal reading log about the MVS depth estimation paper: 
[Y. Yan, et al. MVSNet: Depth Inference for Unstructured Multi-view Stereo, ECCV 2018](https://arxiv.org/abs/1804.02505). 

The note serves mainly for my understanding enhancement but is open to any comments and discussions. I would like to list several keywords here, facilitating my potential memory querying and revisiting in the future.

{mvsnet}, {relative pose}, {homography}, {cost volume}, {sharpness refinement}, {soft argmin}, {supervised}

## Motivation
Measuring the 3D volumetric consistency across multiple views in reference image coordinate. With designed depth, the projected feature volumes from different views observing a common scenario should have less discrepancy conflict, i.e., variance.

## Input
Like normal MVS definition, the input are a reference image $\mathbf{I}_i$, estimating depth for, and several source images { $\mathbf{I}_j: j \in (1, 2,..., N)$ } with viewport variance.

## Prerequisite
Assume the image resolution is $[3, 512, 640]$. Remember to resize intrinsic parameters whenever a down-sampling is applied. Like the vanilla plane-sweeping scheme, the 3D space is uniformly sliced into a sequence of depth samples. For a $[425, 935] mm$ distance and sampling at every $2 mm$, then the depth resolution $D$ would be 256; for $[425, 902.5] mm$, $D = 192$ with stride of $2.5 mm$. So the task is equal to classify each pixel into proper slice.

In order to achieve this, we have to resort to the homography, a.k.a, planar transformation, to establish the bridge between depth and correspondence. A homography is a $3 \times 3$ matrix with $DoF = 8$ and is only good for depicting transformation of the major planar region, leaving other planar layers within the reference image poorly fitted. Pure rotation is its special case which regards the whole image as a plane. However, with depth varying, there would be multiple homography matrices projecting each pixel at different scale to locate which correspondence in source image it should associate to. Note that here backward-projection (from $\mathbf{I}_i$ to $\mathbf{I}_j$) is adopted to ensure matching completeness. Specifically,

```math
x_j = \mathbf{H}_{ji}(d) \cdot x_i, \quad (1)
```

$$
\mathbf{H}_{ji}(d) = \mathbf{K}_j \cdot \mathbf{R}_j \cdot (\mathbf{I} - \frac{(\mathbf{R}^T_j \cdot \mathbf{t}_j - \mathbf{R}^T_i \cdot \mathbf{t}_i) \cdot n^T_i \cdot \mathbf{R}_i}{d}) \cdot \mathbf{R}^T_i \cdot \mathbf{K}^{-1}_i. \quad (2)
$$

Notable that the equation in original paper is incorrect; the listed one is what the authors actually use in code. Normally $n_i$ is the plane normal and $d$ is the offset. Since depth slices are fronto-parallel in our case, so they are equivalent to the principle axis of reference camera and corresponding depth. $[\mathbf{R} | \mathbf{t}]$ is from world-to-camera. To further derivate the equation,

$$
%\begin{equation}
X_j = \mathbf{R}_{ ji } \cdot X_i + \mathbf{t}\_{ji}. \quad (3)
%\end{equation}
$$

Since local 3D points $X_i = d^{-1} \cdot \mathbf{K}^{-1}_i \cdot x_i$ locate on the depth plane, thus

$$
n^T_i \cdot X_i + d = 0, \quad (4)
$$

$$
-\frac{n^T_i \cdot X_i}{d} = 1, \quad (5)
$$

$$
\mathbf{t} = \mathbf{t} \cdot (-\frac{n^T \cdot X}{d}). \quad (6)
$$

Calculating the relative pose, we get

$$
\begin{bmatrix}
\mathbf{R}\_{ji} & \mathbf{t}\_{ji} \\
\mathbf{0}^T & 1 \\
\end{bmatrix}\_{4 \times 4} =
\begin{bmatrix}
\mathbf{R}\_j & \mathbf{t}\_j \\
\mathbf{0}^T & 1 \\
\end{bmatrix}\_{4 \times 4} 
\cdot
\begin{bmatrix}
\mathbf{R}\_i & \mathbf{t}\_j \\
\mathbf{0}^T & 1 \\
\end{bmatrix}^{-1}\_{4 \times 4} =
\begin{bmatrix}
\mathbf{R}\_j & \mathbf{t}\_j \\
\mathbf{0}^T & 1 \\
\end{bmatrix}\_{4 \times 4} \cdot
\begin{bmatrix}
\mathbf{R}^T_i & -\mathbf{R}^T_i \cdot \mathbf{t}\_j \\
\mathbf{0}^T & 1 \\
\end{bmatrix}\_{4 \times 4} =
\begin{bmatrix}
\mathbf{R}\_{ji} & \mathbf{t}\_{ji} \\
\mathbf{0}^T & 1 \\
\end{bmatrix}\_{4 \times 4} =
\begin{bmatrix}
\mathbf{R}\_j \cdot \mathbf{R}^T_i & -\mathbf{R}\_j \cdot \mathbf{R}^T_i \cdot \mathbf{t}\_i + \mathbf{t}\_j \\
\mathbf{0}^T & 1 \\
\end{bmatrix}_{4 \times 4}. \quad (7)
$$

Put them together into Eq.(3),

$$
X_j = \mathbf{R}_j \cdot \mathbf{R}^T_i \cdot X_i - (\mathbf{t}_j - \mathbf{R}_j \cdot \mathbf{R}^T_i \cdot \mathbf{t}_i) \cdot \frac{n^T_i \cdot X_i}{d}, \quad (8)
$$

$$
X_j = \mathbf{R}_j (\mathbf{I} - (\mathbf{R}^T_j \cdot \mathbf{t}_j - \mathbf{R}^T_i \cdot \mathbf{t}_i) \cdot \frac{n^T_i}{d} \cdot \mathbf{R}_i) \cdot \mathbf{R}^T_i \cdot X_i. \quad (9)
$$

Applying the intrinsic projection, it will be in the final form of Eq.(2).

## Feature Extraction
This step uses Conv layers to extract features from each image, i.e., $N + 1$ feature images. The resulting feature map is $1 / 4$ as of the original size in spatial and $32$ in channel. The encoder shares weights across all feature maps, i.e., derived from the same network.

## Feature Volumes
Now we have $[N + 1, 32, 128, 160]$ feature maps. The construction of feature volumes is to warp them, using Eq.(2), into reference coordinate under different depth scales. This will form $D$ new feature maps for each image, i.e., $[N + 1, D, 32, 128, 160]$. Here each $\mathbf{V}_i \in [D, 32, 128, 160]$ corresponds to a feature volume.

## Cost Volume
Since the homography process helps to establish correspondences across different views, the truly matched pixels should have similar features subject to $d$. So by aggregating all feature volumes into one, the final formed volume $\mathbf{C} \in [D, 32, 128, 160]$, a.k.a., cost volume, measures the variance of feature similarity along the first dimension. 

$$
\mathbf{C} = \frac{\sum \limits^{N}_{i=1} (\mathbf{V}_i - \bar{\mathbf{V}})^2}{N}. \quad (10)
$$

$\bar{\mathbf{V}}$ is the average volume among all feature volumes. Think about putting multiple cubic blocks virtually at the same location. Ideally, although there are some regions under-constraint or in improper depth, sufficiently overlapped observations ought to be consistent at designed depth and push down total variance. The cost volume conducting optimizations from a 3D and holistic perspective would be more stable compared to directly regressing geometric knowledge from 2D feature maps. The cost volume representation also makes the algorithm independent of input image numbers.

## Probability Volume
As claimed previously, the depth estimation problem equals to a classification within $D$ depth slices. Therefore, the cost volume has to be further transformed to be able to describe the probability distribution of each pixel in a predefined spatial scope, i.e., $\mathbf{C} \in [D, 32, 128, 160] \stackrel{3D \ Conv}{\longrightarrow} \mathbf{P} \in [D, 1, 128, 160] \stackrel{squeeze}{\longrightarrow} [D, 128, 160]$. Due to the additional $D$ dimension, 3D Conv is required rather than 2D Conv. Consequently, the $softmax$ operation can be applied along the $D$ depth direction for probability normalization.

As the loss is MSE not CE, the winner-take-all $argmax$ is neither able to produce sub-pixel estimation nor differentiable. Instead, taking the $expectation$ could help to resolve these problems

$$
\hat d = \sum \limits^{dmax}_{d = dmin} d \times \mathbf{P}(d). \quad (11)
$$

This operation is also referred to as $soft \ argmin$.

## Sharpness Refinement
As for the common issue of over-smoothness, referring to image matting, a depth residual module is added at the end by incorporating high-frequency information from reference image. Estimated depth map $\hat{\mathbf{d}}$ is concatenated with $\mathbf{I}_i$ as 4-channel input, passing through three 32-channel 2D Conv layers followed by one 1-channel layer to learn the residual, then $\hat{\mathbf{d}}$ is added to get refined depth $\widetilde{\mathbf{d}}$. No BN and ReLU for the last layer. To prevent being biased at a certain depth scale, the initial depth magnitude is scaled to range $[0, 1]$, and converted back after the refinement. The final loss is

$$
Loss = \sum \limits_{p \in \mathbf{P}_{valid}} \| \mathbf{d}(p) - \hat{\mathbf{d}}(p) \|_1 + \lambda \| \mathbf{d}(p) - \widetilde{\mathbf{d}}(p) \|_1, \quad (12)
$$

where $\mathbf{P}_{valid}$ indicates those pixels with valid ground truth.

## Summary
Pros
- From feature maps to feature volumes and cost volume then probability volume, the whole pipeline is intuitive and deserves a careful code reading.

Cons
- Needs 3D Conv, making this kind of methods' mobile deployment intricate.