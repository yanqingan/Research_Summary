**@ 2024 Qingan Yan. Reference or citation is required for re-posting.**

## Preface
A personal reading log about the pioneer diffusion model paper: 
[J. Song, et al. Denoising Diffusion Implicit Models, ICLR 2021](https://arxiv.org/abs/2010.02502). 

The note serves mainly for my understanding enhancement but is open to any comments and discussions. I would like to list several keywords here, facilitating my potential memory querying and revisiting in the future.

{ddim},

## Motivation
DDPM requires a long step-by-step sampling schedule to progressively turn noise into an image, due to its Markovian chain hypothesis, yet DDIM finds that the Markovian chain hypothesis is unnecessary and the sampling can be accelerated by skipping some intervals. So DDIM can be regarded as a more generic format of iterative probabilistic diffusion models.

## DDPM
Recalling the log about [DDPM-NeuIPS20](./DDPM-NeuIPS20.md), the ELBO appears like:
```math
\ln p(x_0) \geq \text{E}_{q(x_{1 : T} | x_0)}[\ln \frac{p_{\theta}(x_0, x_{1 : T})}{q(x_{1 : T} | x_0)}]. \quad (1)
```
$p_{\theta}(x_0, x_{1 : T})$ and $q(x_{1 : T} | x_0)$ correspond to _reverse denoising process_ and _forward diffusion process_ respectively. There are two places involving Markovian chain hypothesis: the 

## DDIM
DDIM shares the same training objective with DDPM, so that there is no need to modify any training pipelines.



## Summary
- Support "short" 
- Consistency