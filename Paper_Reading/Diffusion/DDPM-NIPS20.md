**@2024, Qingan Yan (reference or citation is required for re-posting)**

## Preface
A personal reading log about the pioneer diffusion model paper: 
[Ho et al, Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239). 

The note serves mainly for my understanding enhancement but is open to any comments and discussions. I would like to list several keywords here, facilitating my potential memory querying and revisiting in the future.

{ddpm}

## Motivation
Referring to human behaviors, for instance, in order to assemble a desktop, an efficient way is to practise on some samples by disassembling them into basic accessories step by step (_forward diffusion_). After repeated training on different model types, then giving a series of easily accessible accessories, we will probably know how to assemble them into a powerful desktop (_backward denoising_). Analogously, by watching the degrading procedure from signals to Gaussian distributed noises, diffusion models learn the visual attributes that are important to form a reasonable image, such as semantics, edges and saturation, and their mutual functionalities. So giving an easily accessible Gaussian noise map, diffusion models will try to fuse those attributes into each pixel and generate a visual-friendly image. For generic purposes, diffusion models act as a mapping from easy distribution to target distribution.

## Background
### VAE
The goal of a generative model is to model the true data distribution $p(x)$ as per observed samples $x$ and $y$. GANs learn the distribution in an adversarial manner. Likelihood-based methods seek to learn a model that assigns a high likelihood to the observed data samples. VAEs, besides observed samples, also incorporate with a derived latent variable $z$, assisting the modeling of $p(x) = \int p(x, z)dz$. To achieve this, we have to explicitly marginalize out all latent variables, which is intractable for complex models. Therefore, we appeal to the [chain rule of probability](https://en.wikipedia.org/wiki/Chain_rule_(probability)):
```math
p(x) = \frac{p(x, z)}{p(z | x)}. \quad (1)
```
However, $p(z | x)$ is not accessible. Luckily, it could be approximated by modeling another distribution $q_{\phi}(z | x)$ with parameters $\phi$.

A common way to calculate unknown parameters of a variational distribution is [maximum likelihood estimation](https://en.wikipedia.org/wiki/Maximum_likelihood_estimation), where it tries to maximize the natural logarithm of the likelihood function $p(x)$ (log-likelihood):
```math
\ln p(x) = \ln p(x) \int q_{\phi}(z | x)dz
```
```math
= \int q_{\phi}(z | x)(\ln p(x))dz
```
```math
= \text{E}_{q_{\phi}(z | x)}[\ln p(x)]
```
```math
= \text{E}_{q_{\phi}(z | x)}[\ln \frac{p(x, z)}{p(z | x)}]
```
```math
= \text{E}_{q_{\phi}(z | x)}[\ln \frac{p(x, z)q_{\phi}(z | x)}{p(z | x)q_{\phi}(z | x)}]
```
```math
= \text{E}_{q_{\phi}(z | x)}[\ln \frac{p(x, z)}{q_{\phi}(z | x)}] + \text{E}_{q_{\phi}(z | x)}[\ln \frac{q_{\phi}(x | z)}{p(x | z)}]
```
```math
= \text{E}_{q_{\phi}(z | x)}[\ln \frac{p(x, z)}{q_{\phi}(z | x)}] + D_{KL}(q_{\phi}(x | z) \| p(x | z))
```
```math
\geq \text{E}_{q_{\phi}(z | x)}[\ln \frac{p(x, z)}{q_{\phi}(z | x)}]. \quad (2)
```

Here $D_{KL}$ is a non-negative term and intractable to be minimized, as there is no access to the ground truth $p(z | x)$. By discarding this term, the log-likelihood gets a lower bound Eq.(2), i.e., _ELBO_. So maximizing $\ln p(x)$ is reformulated to increasing the ELBO by tuning the
parameters $\phi$, as in ideal condition the optimized posterior $q_{\phi}(z | x)$ exactly matches the
true posterior $p(z | x)$, leading to $D_{KL} = 0$. However, a problem still remains, i.e., the $p(x, z)$ is unknown. To address this, we can further dissect Eq.(2) via Eq.(1):
```math
\text{E}_{q_{\phi}(z | x)}[\ln \frac{p(x, z)}{q_{\phi}(z | x)}] = \text{E}_{q_{\phi}(z | x)}[\ln \frac{p(x | z)p(z)}{q_{\phi}(z | x)}]
```
```math
= \text{E}_{q_{\phi}(z | x)}[\ln p(x | z) + \ln \frac{p(z)}{q_{\phi}(z | x)}]
```
```math
= \text{E}_{q_{\phi}(z | x)}[\ln p(x | z)] + \text{E}_{q_{\phi}(z | x)}[\ln \frac{p(z)}{q_{\phi}(z | x)}]
```
```math
= \text{E}_{q_{\phi}(z | x)}[\ln p(x | z)] - D_{KL}(q_{\phi}(z | x) \| p(z)). \quad (3)
```
The first term measures the reconstruction likelihood of two distributions. It is easy to imagine that in order to maximize this term, lower probabilities in $p(x | z)$ should also correspond to lower values in $q_{\phi}(z | x)$ and vice versa. This implies that $x$ and $z$ should have distinct association among other potential representations. To simulate the behavior, one feasible solution is to use an _encoder_ network learning the distribution of $q_{\phi}(z | x)$ and another _decoder_ network expressing the distribution of $p_{\theta}(x | z)$. So VAE is a kind of AEs but outputs distribution parameters for feature sampling instead of direct features. Nevertheless, solving the term solely may involve many potential distributions. One commonly used easy proxy is Gaussian distribution. Therefore, we can assume the prior $p(z)$ meets a standard isotropic Gaussian distribution $p(z) \sim N(z; 0, I)$ and $q_{\phi}(z | x)$ ought to be approaching it as much as possible, which is elegantly depicted in the second prior matching term. By simplifying the reconstruction term via a Monte Carlo estimate, the maximization can be rewritten:
```math
\underset{\theta, \phi}{argmax} \, \text{E}_{q_{\phi}(z | x)}[\ln p_{\theta}(x | z)] - D_{KL}(q_{\phi}(z | x) \| p(z)) \approx \underset{\theta, \phi}{argmax} \, \frac{1}{L} \sum \limits^L_{l = 1} \ln p_{\theta}(x | z^l) - D_{KL}(q_{\phi}(z | x) \| p(z)). \quad (4)
```
By taking Gaussian and KL expressions, we can derive a more straightforward format. I will use another log focusing on default VAE to record it. 

Normally, stochastic sampling is non-differentiable, so the reparameterization trick is always adopted when networks output _means_ and _variances_.
```math
z = \mu_{\phi}(x) + \sigma_{phi} \odot \epsilon, \quad \epsilon \sim N(\epsilon; 0, I). \quad (5)
```

### Diffusion
As previously stated, default VAE uses a latent distribution assisting the modeling of $p(x)$, but in practice, interpreting a phenomenon via only one step is hard to draw its essence. A more intuitive way is to profile it gradually until reaching a simple distribution; Each successive step is an abstraction of its ancestor. Therefore, rather than using a single $z$, diffusion equips a sequence of derived latent variables $x_{1 : T}$. According to chain rule of probability and Markovian chain, the distributions used in the ELBO of Eq.(2) will become:
```math
p(x, z) = p(x_0, x_{1 : T}) = p(x_{0 : T}) = p(x_T) \prod \limits^T_{t = 1} p_{\theta}(x_{t - 1} | x_t), \quad p(x_T) \sim N(x_T; 0, I), \quad (6)
```
```math
q(z | x) = q(x_{1 : T} | x_0) = \prod \limits^T_{t = 1} q(x_t | x_{t - 1}), \quad (7)
``` 
which correspond to _reverse denoising process_ and _forward diffusion process_ respectively.

Another important distinction is the generation of latent distributions. In contrast to VAEs which use a network to model the procedure, the generation in diffusion is pre-scheduled, i.e., adding noises with known means and variances at each timestep. So there are no parameters for $q(z | x)$. Specifically, the noise scheduler is based on previous status with some linear perturbation, like $y = ax + b$:
```math
q(x_t | x_{t - 1}) = N(x_t; \sqrt{\alpha_t}x_{t - 1}, \beta_t I).
```
Many diffusion models want at first the perturbation to be small but increase as time goes by. This inspires _variance-preserving_ scheduler, i.e., $\alpha_t + \beta_t = 1$, which retains more information at the beginning yet collapses faster and faster later:
```math
q(x_t | x_{t - 1}) = N(x_t; \sqrt{1 - \beta_t}x_{t - 1}, \beta_t I). \quad (8)
```
The scheduler can also be learned or act in a _variance-exploding_ manner. I will try to cover them with corresponding papers in other logs.

A good property of ...

## Summary
Pros
1. 
Cons
1. Slow.
2. 
