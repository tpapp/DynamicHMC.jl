# [Introduction](@id Introduction)

This package implements a variant of the “No-U-Turn Sampler” of Hoffmann and [Gelman (2014)](https://arxiv.org/abs/1111.4246), as described in [Betancourt (2017)](https://arxiv.org/abs/1701.02434). **In order to make the best use of this package, you should read at least the latter paper thoroughly**.

This package is mainly useful for Bayesian inference. To make the best use of it, you need to be familiar with the conceptual building blocks of Bayesian inference, most importantly, you should be able to code a posterior function in Julia. For various techniques and a discussion of MCMC methods, you may the [Stan Modeling Language manual](http://mc-stan.org/users/documentation/index.html) helpful. If you are unfamiliar with Bayesian methods, [Bayesian Data Analysis](http://www.stat.columbia.edu/~gelman/book/) is a good introduction, among other great books.

The package aims to “[do one thing and do it well](https://en.wikipedia.org/wiki/Unix_philosophy#Do_One_Thing_and_Do_It_Well)”: given a log density function

```math
\ell: \mathbb{R}^k \to \mathbb{R}
```

for which you have values ``\ell(x)`` and the gradient ``\nabla \ell(x)``, it samples values from a density

```math
p(x) \propto \exp(\ell(x))
```

using the algorithm above.

The package provides a framework to [tune](@ref tuning) the algorithm to find near-optimal parameters for sampling, and also [diagnostics](@ref Diagnostics) that are specific to the algorithm.

However, following a modular approach, it does *not* provide

1. Domain transformations from subsets of ``\mathbb{R}^k``. For that, see [ContinuousTransformations.jl](https://github.com/tpapp/ContinuousTransformations.jl).

2. Automatic differentiation. Julia has a thriving [AD ecosystem](http://www.juliadiff.org/) which should allow you to implement this. [DiffWrappers.jl](https://github.com/tpapp/DiffWrappers.jl) should automate this in a single line.

3. Generic MCMC diagnostics not specific to NUTS. See [MCMCDiagnostics.jl](https://github.com/tpapp/MCMCDiagnostics.jl) for an implementation of ``\hat{R}`` and effective sample size calculations.
