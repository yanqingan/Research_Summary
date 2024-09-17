**@ 2024 Qingan Yan. Reference or citation is required for re-posting.**

## Preface
A personal reading log about the pioneer diffusion model paper: 
[J. Ho, et al. Denoising Diffusion Probabilistic Models, NeuIPS 2020](https://arxiv.org/abs/2006.11239). 

The note serves mainly for my understanding enhancement but is open to any comments and discussions. I would like to list several keywords here, facilitating my potential memory querying and revisiting in the future.

{ddpm}, {vae}, {elbo}, {maximum likelihood estimation}, {monte carlo}, {variance preserving}, {kl divergence}

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
= \text{E}_{q_{\phi}(z | x)}[\ln \frac{p(x, z)}{q_{\phi}(z | x)}] + \text{D}_{KL}(q_{\phi}(x | z) \| p(x | z))
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
= \text{E}_{q_{\phi}(z | x)}[\ln p(x | z)] - \text{D}_{KL}(q_{\phi}(z | x) \| p(z)). \quad (3)
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
q(x_t | x_{t - 1}) = N(x_t; \sqrt{\alpha_t} x_{t - 1}, \beta_t I).
```
Many diffusion models want at first the perturbation to be small but increase as time goes by. This inspires _variance-preserving_ scheduler, i.e., $\alpha_t + \beta_t = 1$, which retains more information at the beginning yet collapses faster and faster later:
```math
q(x_t | x_{t - 1}) = N(x_t; \sqrt{1 - \beta_t} x_{t - 1}, \beta_t I). \quad (8)
```
The scheduler can also be learned or act in a _variance-exploding_ manner. I will try to cover them with corresponding papers in other logs.

A good property of the predefined Gaussian scheduler is that it enables to locate sequent distributions from the beginning. Use the reparameterization equation:
```math
x_t = \sqrt{\alpha_t} x_{t - 1} + \sqrt{1 - \alpha_t} \epsilon_{t - 1}
```
```math
= \sqrt{\alpha_t} (\sqrt{\alpha_{t - 1}} x_{t - 2} + \sqrt{1 - \alpha_{t - 1}} \epsilon_{t - 2}) + \sqrt{1 - \alpha_t} \epsilon_{t - 1}
```
```math
= \sqrt{\alpha_t \alpha_{t - 1}} x_{t - 2} + \sqrt{\alpha_t - \alpha_t \alpha_{t - 1}} \epsilon_{t - 2} + \sqrt{1 - \alpha_t} \epsilon_{t - 1}.
```
Since the sum of two independent Gaussian random variables remains a Gaussian with mean being the sum of the two means, and variance being the sum of the two variances, the equation further goes to:
```math
x_t = \sqrt{\alpha_t \alpha_{t - 1}} x_{t - 2} + \sqrt{(\alpha_t - \alpha_t \alpha_{t - 1}) + (1 - \alpha_t)} \hat{\epsilon}_{t - 2}
```
```math
= \sqrt{\alpha_t \alpha_{t - 1}} x_{t - 2} + \sqrt{1 - \alpha_t \alpha_{t - 1}} \hat{\epsilon}_{t - 2}
```
```math
= \dots
```
```math
= \sqrt{\prod \limits^t_{i = 1} \alpha_i} x_0 + \sqrt{1 - \prod \limits^t_{i = 1} \alpha_i} \hat{\epsilon}_0
```
```math
= \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \hat{\epsilon}_0, \quad \bar{\alpha}_t = \prod \limits^t_{i = 1} \alpha_i
```
```math
\sim N(x_t; \sqrt{\bar{\alpha}_t} x_0, (1 - \bar{\alpha}_t) I). \quad (9)
```

Let's recall the ELBO of Eq.(2); in terms of diffusion models, it reformulates to:
```math
\text{E}_{q_{\phi}(z | x)}[\ln \frac{p(x, z)}{q_{\phi}(z | x)}] = \text{E}_{q(x_{1 : T} | x_0)}[\ln \frac{p(x_0, x_{1 : T})}{q(x_{1 : T} | x_0)}] 
```
```math
= \text{E}_{q(x_{1 : T} | x_0)}[\ln \frac{p(x_T) \prod \limits^T_{t = 1} p_{\theta}(x_{t - 1} | x_t)}{\prod \limits^T_{t = 1} q(x_t | x_{t - 1})}] \quad (10)
```
```math
= \text{E}_{q(x_{1 : T} | x_0)}[\ln \frac{p(x_T) p_{\theta}(x_0 | x_1)
\prod \limits^T_{t = 2} p_{\theta}(x_{t - 1} | x_t)}
{q(x_T | x_{T - 1}) \prod \limits^{T - 1}_{t = 1} q(x_t | x_{t - 1})}]
```
```math
= \text{E}_{q(x_{1 : T} | x_0)}[\ln \frac{p(x_T) p_
{\theta}(x_0 | x_1)
\prod \limits^{T - 1}_{t = 1} p_{\theta}(x_{t} | x_{t + 1})}
{q(x_T | x_{T - 1}) \prod \limits^{T - 1}_{t = 1} q
(x_t | x_{t - 1})}] \quad (11)
```
```math
= \text{E}_{q(x_{1 : T} | x_0)}[\ln \frac{p(x_T) p_{\theta}(x_0 | x_1)}{q(x_T | x_{T - 1})}]
+ \text{E}_{q(x_{1 : T} | x_0)}[\ln \prod \limits^{T - 1}_{t = 1} \frac{ p_{\theta}(x_t | x_{t + 1})}{q(x_t | x_{t - 1})}]
```
```math
= \text{E}_{q(x_{1 : T} | x_0)}[\ln p_{\theta}(x_0 | x_1)]
+ \text{E}_{q(x_{1 : T} | x_0)}[\ln \frac{p(x_T)}{q(x_T | x_{T - 1})}]
+ \sum \limits^{T - 1}_{t = 1} \text{E}_{q(x_{1 : T} | x_0)}[\ln \frac{ p_{\theta}(x_t | x_{t + 1})}{q(x_t | x_{t - 1})}]
```
```math
= \text{E}_{q(x_1 | x_0)}[\ln p_{\theta}(x_0 | x_1)]
+ \text{E}_{q(x_{T - 1}, x_T | x_0)}[\ln \frac{p(x_T)}{q(x_T | x_{T - 1})}]
+ \sum \limits^{T - 1}_{t = 1} \text{E}_{q(x_{t - 1}, x_t, x_{t + 1} | x_0)}[\ln \frac{p_{\theta}(x_t | x_{t + 1})}{q(x_t | x_{t - 1})}] \quad (12)
```
```math
= \text{E}_{q(x_1 | x_0)}[\ln p_{\theta}(x_0 | x_1)]
- \text{E}_{q(x_{T - 1} | x_0)}[\text{D}_{KL}(q(x_T | x_{T - 1}) \| p(x_T))]
- \sum \limits^{T - 1}_{t = 1} \text{E}_{q(x_{t - 1}, x_{t + 1} | x_0)}[\text{D}_{KL}(q(x_t | x_{t - 1}) \| p_{\theta}(x_t | x_{t + 1}))]. \quad (13)
```
So this is the initial expanded formulation of ELBO in likelihood diffusion models. For the transmit between Eq.(12) and Eq.(13), it can be derived by expanding $q(x_{T - 1}, x_T | x_0)$ and $q(x_{t - 1}, x_t, x_{t + 1} | x_0)$ into $q(x_{T - 1} | x_0) q(x_T | x_{T - 1})$ and $q(x_{t - 1}, x_{t + 1} | x_0) q(x_t | x_{t - 1})$.

The first term is a _reconstruction_ term, predicting the log similarity of original data and its one-step latent, which also appears in vanilla VAEs. The second term is named as _prior matching_ term, which ensure that the final latent distribution matches the Gaussian prior. Since there are no learnable parameters, and with a large enough timestep $T$ the divergence is a constant approaching zero, therefore the term can be ignored. However, later in other articles, we will see that in training the final distribution is not guaranteed to be an isotropic Gaussian, causing some bias in generating totally dark or bright images. The third term is the _consistency_ term which endeavors to keep distributions from noising and denoising processes consistent. Notice that there are some issues for the consistency term. First, it is inconvenient to deal with both $x_{t + 1}$ and $x_{t - 1}$. Second, the ELBO might be suboptimal as variance of Monte Carlo estimate for expectation would be higher on two random variables than on only one.

So revisiting Eq.(11), the $x_{t + 1}$ appears after adjusting the index which aims to make the same $x_t$ in $p_{\theta}$ and $q$. This time we resort to $q(x_t | x_{t - 1}) = q(x_t | x_{t - 1}, x_0)$ as the Markov property and the three element Bayes rule:
```math
q(x_t | x_{t - 1}, x_0) = \frac{q(x_{t - 1} | x_t, x_0)q(x_t | x_0)}{q(x_{t - 1} | x_0)}. \quad (14)
```
Restart from E.q(10) where:
```math
\text{E}_{q(x_{1 : T} | x_0)}[\ln \frac{p(x_T) 
\prod \limits^T_{t = 1} p_{\theta}(x_{t - 1} | x_t)}
{\prod \limits^T_{t = 1} q(x_t | x_{t - 1})}] \quad 
(10)
```
```math
= \text{E}_{q(x_{1 : T} | x_0)}[\ln \frac{p(x_T) p_{\theta}(x_0 | x_1) \prod \limits^T_{t = 2} p_{\theta}(x_{t - 1} | x_t)}{q(x_1 | x_0) \prod \limits^T_{t = 2} q(x_t | x_{t - 1})}]
```
```math
= \text{E}_{q(x_{1 : T} | x_0)}[\ln \frac{p(x_T) p_{\theta}(x_0 | x_1)}{q(x_1 | x_0)} 
+ \ln \prod \limits^T_{t = 2} \frac{p_{\theta}(x_{t - 1} | x_t)}{q(x_t | x_{t - 1}, x_0)}]
```
```math
= \text{E}_{q(x_{1 : T} | x_0)}[\ln \frac{p(x_T) p_{\theta}(x_0 | x_1)}{q(x_1 | x_0)} 
+ \ln \prod \limits^T_{t = 2} \frac{p_{\theta}(x_{t - 1} | x_t)}{\frac{q(x_{t - 1} | x_t, x_0)q(x_t | x_0)}{q(x_{t - 1} | x_0)}}]
```
```math
= \text{E}_{q(x_{1 : T} | x_0)}[\ln \frac{p(x_T) p_{\theta}(x_0 | x_1)}{q(x_1 | x_0)} 
+ \ln (\dots \frac{p_{\theta}(x_{t - 1} | x_t)}{\frac{q(x_{t - 1} | x_t, x_0)q(x_t | x_0)}{q(x_{t - 1} | x_0)}} \frac{p_{\theta}(x_t | x_{t + 1})}{\frac{q(x_t | x_{t + 1}, x_0)q(x_{t + 1} | x_0)}{q(x_t | x_0)}} \dots)]
```
```math
= \text{E}_{q(x_{1 : T} | x_0)}[\ln \frac{p(x_T) p_{\theta}(x_0 | x_1)}{q(x_1 | x_0)} 
+ \ln \frac{q(x_1 | x_0)}{q(x_T | x_0)} 
+ \ln \prod \limits^T_{t = 2} \frac{p_{\theta}(x_{t - 1} | x_t)}{q(x_{t - 1} | x_t, x_0)}]
```
```math
= \text{E}_{q(x_{1 : T} | x_0)}[\ln \frac{p(x_T) p_{\theta}(x_0 | x_1)}{q(x_T | x_0)} + \ln \prod \limits^T_{t = 2} \frac{p_{\theta}(x_{t - 1} | x_t)}{q(x_{t - 1} | x_t, x_0)}]
```
```math
= \text{E}_{q(x_{1 : T} | x_0)}[\ln p_{\theta}(x_0 | x_1)]
+ \text{E}_{q(x_{1 : T} | x_0)}[\ln \frac{p(x_T)}{q(x_T | x_0)}]
+ \sum \limits^T_{t = 2} \text{E}_{q(x_{1 : T} | x_0)}[\ln \frac{ p_{\theta}(x_{t - 1} | x_t)}{q(x_{t - 1} | x_t, x_0)}]
```
```math
= \text{E}_{q(x_1 | x_0)}[\ln p_{\theta}(x_0 | x_1)]
+ \text{E}_{q(x_T | x_0)}[\ln \frac{p(x_T)}{q(x_T | x_0)}]
+ \sum \limits^T_{t = 2} \text{E}_{q(x_t, x_{t - 1} | x_0)}[\ln \frac{ p_{\theta}(x_{t - 1} | x_t)}{q(x_{t - 1} | x_t, x_0)}]
```
```math
= \text{E}_{q(x_1 | x_0)}[\ln p_{\theta}(x_0 | x_1)]
+ \text{E}_{q(x_T | x_0)}[\ln \frac{p(x_T)}{q(x_T | x_0)}]
+ \sum \limits^T_{t = 2} \text{E}_{q(x_{t - 1} | x_t, x_0) q(x_t | x_0)}[\ln \frac{ p_{\theta}(x_{t - 1} | x_t)}{q(x_{t - 1} | x_t, x_0)}]
```
```math
= \text{E}_{q(x_1 | x_0)}[\ln p_{\theta}(x_0 | x_1)]
- \text{D}_{KL}[q(x_T | x_0) \| p(x_T)]
- \sum \limits^T_{t = 2} \text{E}_{q(x_t | x_0)}[D_{KL}(q(x_{t - 1} | x_t, x_0) \| p_{\theta}(x_{t - 1} | x_t))]. \quad (15)
```

Now the new ELBO only relates to one random variable at each timestep. Here I would like to talk more about how to optimize each term:
1. The _reconstruction_ term measures the (MSE) similarity between $x_0$ and estimated mean $\mu^0_{\theta}$. Since it is the first step and very rare noises are imposed, i.e., $\alpha_0 \approx 1.0$ and $\beta_0 \approx 0.0$, thus $\mu^0_{\theta}$ can be directly regarded as an estimate of $x_0$ from decoder $\theta$. The expectation over $q(x_1 | x_0)$ can be approximated via Monte Carlo estimate. Go with equations to carve it deeply:
```math
\text{E}_{q(x_1 | x_0)}[\ln p_{\theta}(x_0 | x_1)] \approx \frac{1}{L} \sum \limits^L_{l = 1} \ln p_{\theta}(x_0 | x_1)
```
```math
= \frac{1}{L} \sum \limits^L_{l = 1} \ln \frac{1}{\sqrt{2 \pi} \sigma} \exp(-\frac{(x_0 - \mu_{\theta}(x_1, 1))^2}{2 \sigma^2})
```
```math
= \frac{1}{L} \sum \limits^L_{l = 1} \ln \frac{1}{\sqrt{2 \pi} \sigma} + \ln \exp(-\frac{(x_0 - \mu_{\theta}(x_1, 1))^2}{2 \sigma^2})
```
```math
\approx \frac{1}{L} \sum \limits^L_{l = 1} (-\frac{(x_0 - \mu_{\theta}(x_1, 1))^2}{2 \sigma^2})
```
```math
\propto -\frac{1}{L} \sum \limits^L_{l = 1} \| x_0 - \mu_{\theta}(x_1, 1) \|^2_2. \quad (16)
```
2. The _prior matching_ term has no learnable parameters and with enough steps the final distribution would match to isotropic Gaussian. So it is usually ignored in optimization.
3. The third term becomes to a _denoising matching_ term where it hopes the two denoising distributions match closely. In order to optimize it, first have to compute $q(x_{t - 1} | x_t, x_0)$. We know via Eq.(14) its Bayesian format and $q(x_t | x_{t - 1}, x_0) = q(x_t | x_{t - 1})$ as Markov property, $q(x_t | x_0)$ and $q(x_{t - 1} | x_0)$ are accessible using Eq.(9).
```math
q(x_{t - 1} | x_t, x_0) = \frac{q(x_t | x_{t - 1}, x_0) q(x_{t - 1} | x_0)}{q(x_t | x_0)}
```
```math
= \frac{N(x_t; \sqrt{\alpha_t} x_{t - 1}, (1 - \alpha_t)I) \ N(x_{t - 1}; \sqrt{\bar{\alpha}_{t - 1}} x_0, (1 - \bar{\alpha}_{t - 1})I)}{N(x_t; \sqrt{\bar{\alpha}_t} x_0, (1 - \bar{\alpha}_t)I)}
```
```math
\propto \exp(-\frac{1}{2} (\frac{(x_t - \sqrt{\alpha_t} x_{t - 1})^2}{1 - \alpha_t} 
+ \frac{(x_{t - 1} - \sqrt{\bar{\alpha}_{t - 1}} x_0)^2}{1 - \bar{\alpha}_{t - 1}}
- \frac{(x_t - \sqrt{\bar{\alpha}_t} x_0)^2}{1 - \bar{\alpha}_t}))
```
```math
= \exp(-\frac{1}{2} (\frac{-2 \sqrt{\alpha_t} x_t x_{t - 1} + \alpha_t x^2_{t - 1}}{1 - \alpha_t} 
+ \frac{x^2_{t - 1} - 2 \sqrt{\bar{\alpha}_{t - 1}} x_{t - 1} x_0}{1 - \bar{\alpha}_{t - 1}}
+ C(x_t, x_0)))
```
```math
\propto \exp(-\frac{1}{2} (-\frac{2 \sqrt{\alpha_t} x_t x_{t - 1}}{1 - \alpha_t} + \frac{\alpha_t x^2_{t - 1}}{1 - \alpha_t} 
+ \frac{x^2_{t - 1}}{1 - \bar{\alpha}_{t - 1}} - \frac{2 \sqrt{\bar{\alpha}_{t - 1}} x_{t - 1} x_0}{1 - \bar{\alpha}_{t - 1}}))
```
```math
= \exp(-\frac{1}{2} (\frac{\alpha_t x^2_{t - 1}}{1 - \alpha_t} + \frac{x^2_{t - 1}}{1 - \bar{\alpha}_{t - 1}} 
- \frac{2 \sqrt{\alpha_t} x_t x_{t - 1}}{1 - \alpha_t} - \frac{2 \sqrt{\bar{\alpha}_{t - 1}} x_{t - 1} x_0}{1 - \bar{\alpha}_{t - 1}}))
```
```math
= \exp(-\frac{1}{2} ((\frac{\alpha_t}{1 - \alpha_t} + \frac{1}{1 - \bar{\alpha}_{t - 1}}) x^2_{t - 1}
- 2 (\frac{\sqrt{\alpha_t} x_t}{1 - \alpha_t} - \frac{\sqrt{\bar{\alpha}_{t - 1}} x_0}{1 - \bar{\alpha}_{t - 1}}) x_{t - 1}))
```
```math
= \exp(-\frac{1}{2} (\frac{1 - \bar{\alpha}_t}{(1 - \alpha_t)(1 - \bar{\alpha}_{t - 1})} x^2_{t - 1}
- 2 (\frac{\sqrt{\alpha_t} x_t}{1 - \alpha_t} - \frac{\sqrt{\bar{\alpha}_{t - 1}} x_0}{1 - \bar{\alpha}_{t - 1}}) x_{t - 1}))
```
```math
= \exp(-\frac{1}{2} \frac{1 - \bar{\alpha}_t}{(1 - \alpha_t)(1 - \bar{\alpha}_{t - 1})} 
(x^2_{t - 1} - 2 \frac{(\frac{\sqrt{\alpha_t} x_t}{1 - \alpha_t} - \frac{\sqrt{\bar{\alpha}_{t - 1}} x_0}{1 - \bar{\alpha}_{t - 1}})}{\frac{1 - \bar{\alpha}_t}{(1 - \alpha_t) (1 - \bar{\alpha}_{t - 1})}} x_{t - 1}))
```
```math
= \exp(-\frac{1}{2} (\frac{1}{\frac{(1 - \alpha_t) (1 - \bar{\alpha}_{t - 1})}{1 - \bar{\alpha}_t}}) 
(x^2_{t - 1} - 2 \frac{\sqrt{\alpha_t} (1 - \bar{\alpha}_{t - 1}) x_t + \sqrt{\bar{\alpha}_{t - 1}} (1 - \alpha_t) x_0}{1 - \bar{\alpha}_t} x_{t - 1}))
```
```math
\propto N(x_{t - 1}; \frac{\sqrt{\alpha_t} (1 - \bar{\alpha}_{t - 1}) x_t + \sqrt{\bar{\alpha}_{t - 1}} (1 - \alpha_t) x_0}{1 - \bar{\alpha}_t}, \frac{(1 - \alpha_t) (1 - \bar{\alpha}_{t - 1})}{1 - \bar{\alpha}_t}) \quad (17)
```
3. Accordingly, to optimize the KL divergence in Eq.(15) equals to make the mean and variance of $p_{\theta}(x_{t - 1} | x_t)$ as close as possible to $q(x_{t - 1} | x_t, x_0)$. The theoretical derivation relies on the mathematical formula of KL divergence, I might catch up with it later if have chances, but intuitively we can note that since the variance in Eq.(17) relates only to a predefined scheduler $\alpha$, so the optimization just cares about the mean. For further simplification, subject to non-learnable schedulers, we can be safe to ignore the coefficient, focusing only on the content in norm:
```math
\sum \limits^T_{t = 2} \text{E}_{q(x_t | x_0)}[D_{KL}(q(x_{t - 1} | x_t, x_0) \| p_{\theta}(x_{t - 1} | x_t))] \approx \frac{1}{L} \sum \limits^T_{t = 2} \sum \limits^L_{l = 1} \| \mu_{\theta}(x_t, t) - \mu_q(x_t, x_0) \|^2_2 \quad (18)
```
```math
= \frac{1}{L} \sum \limits^T_{t = 2} \sum \limits^L_{l = 1} \| \frac{\sqrt{\alpha_t} (1 - \bar{\alpha}_{t - 1}) x_t + \sqrt{\bar{\alpha}_{t - 1}} (1 - \alpha_t) x_{\theta}(x_t, t)}{1 - \bar{\alpha}_t} 
- \frac{\sqrt{\alpha_t} (1 - \bar{\alpha}_{t - 1}) x_t + \sqrt{\bar{\alpha}_{t - 1}} (1 - \alpha_t) x_0}{1 - \bar{\alpha}_t} \|^2_2
```
```math
\propto \frac{1}{L} \sum \limits^T_{t = 2} \sum \limits^L_{l = 1} \| x_{\theta}(x_t, t) - x_0 \|^2_2. \quad (19)
```

## DDPM
Now we have plainly go over the likelihood diffusion theory, which corresponds to Sec.2 in original paper. As stated previously, there are two ways to regularize the KL divergence: (1) Directly estimate means, but it is unstable to train in practice; (2) Recover $x_0$ giving timestep $t$ and its noisy image $x_t$. In contrast, DDPM chooses to optimize on noises and achieve better performance. 

Furthermore, by using the inverse format of Eq.(9)
```math
x_0 = \frac{x_t - \sqrt{1 - \bar{\alpha}_t} \epsilon_0}{\sqrt{\bar{\alpha}_t}},
```
the mean can be simplified into
```math
\mu_q(x_t, x_0) = \frac{\sqrt{\alpha_t} (1 - \bar{\alpha}_{t - 1}) x_t + \sqrt{\bar{\alpha}_{t - 1}} (1 - \alpha_t) \frac{x_t - \sqrt{1 - \bar{\alpha}_t} \epsilon_0}{\sqrt{\bar{\alpha}_t}}}{1 - \bar{\alpha}_t}
```
```math
= \frac{1}{\sqrt{\alpha_t}} x_t - \frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha}_t} \sqrt{\alpha_t}} \epsilon_0. \quad (20)
```
Subsequently, the denoising matching term turns Eq.(18) or Eq.(19) to approximate noises that are imposed on $x_0$
```math
\frac{1}{L} \sum \limits^T_{t = 2} \sum \limits^L_{l = 1} \| \mu_{\theta}(x_t, t) - \mu_q(x_t, x_0) \|^2_2 = \frac{1}{L} \sum \limits^T_{t = 2} \sum \limits^L_{l = 1} \| \frac{1}{\sqrt{\alpha_t}} x_t - \frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha}_t} \sqrt{\alpha_t}} \epsilon_{\theta}(x_t, t) - \frac{1}{\sqrt{\alpha_t}} x_t + \frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha}_t} \sqrt{\alpha_t}} \epsilon_0 \|^2_2
```
```math
\propto \frac{1}{L} \sum \limits^T_{t = 2} \sum \limits^L_{l = 1} \| \epsilon_0 - \epsilon_{\theta}(x_t, t) \|^2_2. \quad (21)
```

Here have to very careful about the the compute of $x_t$. As Eq.(21) is only used in training, $x_t$ can be directly located via the reparameterization trick Eq.(9) on $x_0$. However, at inference/sampling time, we have to input $x_t$ into networks to estimate $x_{t - 1}$ without the basis of $x_0$. In this case, Eq.(20) is leveraged to calculate the mean and Eq.(17) is applied to sample $x_t$.

Overall, because of $\bar{\alpha}_1$ = $\alpha_1$, the simplified training loss (discards the weighting) will be the in a concise format, combining Eq.(16) and Eq.(21) in accordance with the sign of Eq.(15):
```math
\underset{\theta}{argmin} \, L_{simple}(\theta) = -(-\frac{1}{L} \sum \limits^L_{l = 1} \| x_0 - \mu_{\theta}(x_1, 1) \|^2_2 - \frac{1}{L} \sum \limits^T_{t = 2} \sum \limits^L_{l = 1} \| \epsilon_0 - \epsilon_{\theta}(x_t, t) \|^2_2)
```
```math
= \frac{1}{L} \sum \limits^L_{l = 1} \| x_0 - \mu_{\theta}(x_1, 1) \|^2_2 + \frac{1}{L} \sum \limits^T_{t = 2} \sum \limits^L_{l = 1} \| \epsilon_0 - \epsilon_{\theta}(x_t, t) \|^2_2
```
```math
= \frac{1}{L} \sum \limits^L_{l = 1} \| \frac{x_1 - \sqrt{1 - \bar{\alpha}_1} \epsilon_0}{\sqrt{\bar{\alpha}_1}} - \frac{1}{\sqrt{\alpha_1}} x_1 + \frac{1 - \alpha_1}{\sqrt{1 - \bar{\alpha}_1} \sqrt{\alpha_1}} \epsilon_{\theta}(x_1, 1) \|^2_2 + \frac{1}{L} \sum \limits^T_{t = 2} \sum \limits^L_{l = 1} \| \epsilon_0 - \epsilon_{\theta}(x_t, t) \|^2_2
```
```math
= \frac{1}{L} \sum \limits^L_{l = 1} \| -\frac{\sqrt{1 - \alpha_1}}{\sqrt{\alpha_1}} \epsilon_0 + \frac{1 - \alpha_1}{\sqrt{1 - \alpha_1} \sqrt{\alpha_1}} \epsilon_{\theta}(x_1, 1) \|^2_2 + \frac{1}{L} \sum \limits^T_{t = 2} \sum \limits^L_{l = 1} \| \epsilon_0 - \epsilon_{\theta}(x_t, t) \|^2_2
```
```math
\approx \frac{1}{L} \sum \limits^L_{l = 1} \| \epsilon_{\theta}(x_1, 1) - \epsilon_0 \|^2_2 + \frac{1}{L} \sum \limits^T_{t = 2} \sum \limits^L_{l = 1} \| \epsilon_0 - \epsilon_{\theta}(x_t, t) \|^2_2 \quad (22)
```
```math
= \frac{1}{L} \sum \limits^T_{t = 1} \sum \limits^L_{l = 1} \| \epsilon_0 - \epsilon_{\theta}(x_t, t) \|^2_2. \quad (23)
```

In inference, in contrast to the other steps, the last transmit directly utilizes the estimated mean 
```math
\frac{1}{\sqrt{\alpha_1}} x_1 - \frac{1 - \alpha_1}{\sqrt{1 - \bar{\alpha}_1} \sqrt{\alpha_1}} \epsilon_{\theta}(x_1, 1) \quad (24)
``` 
as final output $\hat{x}_0$ rather than drawing a sample from the distribution. This is because: (1) It is regularized in the reconstruction term; (2) It can improve the determinism as the last step has very low variance.

## Summary
1. Due to the Markovian chain hypothesis, DDPM prefers the transmit to be approximately contiguous. So it requires a long and cosy sampling path, usually $1000$ steps, in order to get a generative output.
2. DDPM ignores the weighting in ELBO or the simplified loss function. So that by drawing from Monte Carlo estimate it treats each step uniformly, although theoretically some steps should be downgraded or upgraded.
3. The scheduler $\beta$ varies from $\beta_1 = 10^{-4}$ to $\beta_T = 0.02$, which actually does not make the last distribution a truly standard Gaussian. Therefore, it and many of its following variants can only generate plain medium brightness images.
4. While can produce high-quality and diverse images, vanilla DDPM does not support resolution changes or conditional guidance.

## References
1. [K. Luo. Understanding Diffusion Models: A Unified Perspective, 2022](https://arxiv.org/abs/2208.11970).