# Introduction

DynamicHMC.jl implements a variant of the “No-U-Turn Sampler” of Hoffmann and [Gelman (2014)](https://arxiv.org/abs/1111.4246), as described in [Betancourt (2017)](https://arxiv.org/abs/1701.02434).[^1] This package is mainly useful for Bayesian inference.

[^1]: In order to make the best use of this package, you should read at least the latter paper thoroughly.

In order to use it, you should be familiar with the conceptual building blocks of Bayesian inference, most importantly, you should be able to code a (log) posterior as a function in Julia.[^2] The package aims to “[do one thing and do it well](https://en.wikipedia.org/wiki/Unix_philosophy#Do_One_Thing_and_Do_It_Well)”: given a log density function

```math
\ell: \mathbb{R}^k \to \mathbb{R}
```

for which you have values ``\ell(x)`` and the gradient ``\nabla \ell(x)``, it samples values from a density

```math
p(x) \propto \exp(\ell(x))
```

using the algorithm above.

[^2]: For various techniques and a discussion of MCMC methods (eg domain transformations, or integrating out discrete parameters), you may find the [Stan Modeling Language manual](http://mc-stan.org/users/documentation/index.html) helpful. If you are unfamiliar with Bayesian methods, I would recommend [Bayesian Data Analysis](http://www.stat.columbia.edu/~gelman/book/) and [Statistical Rethinking](https://xcelab.net/rm/statistical-rethinking/).

The interface of DynamicHMC.jl expects that you code ``\ell(x), \nabla\ell(x)`` using the interface of the [LogDensityProblems.jl](https://github.com/tpapp/LogDensityProblems.jl) package. The latter package also allows you to just code ``\ell(x)`` and obtain ``\nabla\ell(x)`` via [automatic differentiation](https://en.wikipedia.org/wiki/Automatic_differentiation).

While the NUTS algorithm operates on an *unrestricted* domain ``\mathbb{R}^k``, some parameters have natural restrictions: for example, standard deviations are positive, valid correlation matrices are a subset of all matrices, and structural econometric models can have parameter restrictions for stability. In order to sample for posteriors with parameters like these, *domain transformations* are required.[^3] Also, it is advantageous to decompose a flat vector `x` to a collection of parameters in a disciplined manner.

[^3]: For nonlinear transformations, correcting with the logarithm of the determinant of the Jacobian is required.

I recommend that you use [TransformVariables.jl](https://github.com/tpapp/TransformVariables.jl) in combination with LogdensityProblems.jl for this purpose: it has built-in transformations for common cases, and also allows decomposing vectors into tuples, named tuples, and arrays of parameters, combined with these transformations.

## Use cases

This package has the following intended use cases:

1. A robust and simple engine for MCMC. The intended audience is users who like to code their (log)posteriors directly, optimize and benchmark them them as Julia code, and at the same time want to have access detailed diagnostic information from the NUTS sampler.

2. A *backend* for another interface that needs a NUTS implementation.

3. A *research platform* for advances in MCMC methods. The code of this package is extensively documented, and should allow extensions and experiments easily using multiple dispatch. Contributions are always welcome.

## Support

If you have questions, feature requests, or bug reports, please [open an issue](https://github.com/tpapp/DynamicHMC.jl/issues/new). I would like to emphasize that it is perfectly fine to open issues just to ask questions. You can also address questions to [@Tamas_Papp](https://discourse.julialang.org/u/Tamas_Papp) on the Julia discourse forum.
