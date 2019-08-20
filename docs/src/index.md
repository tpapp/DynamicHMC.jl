# Tutorial

## Introduction

DynamicHMC.jl is an implements a variant of the “No-U-Turn Sampler” of Hoffmann and [Gelman (2014)](https://arxiv.org/abs/1111.4246), as described in [Betancourt (2017)](https://arxiv.org/abs/1701.02434).[^Betancourt2017] This package is mainly useful for Bayesian inference.

[^Betancourt2017]: In order to make the best use of this package, you should read at least the latter paper thoroughly.

In order to use it, you need to be familiar with the conceptual building blocks of Bayesian inference, most importantly, you should be able to code a (log) posterior as a function in Julia.[^MCMCintro] The package aims to “[do one thing and do it well](https://en.wikipedia.org/wiki/Unix_philosophy#Do_One_Thing_and_Do_It_Well)”: given a log density function

```math
\ell: \mathbb{R}^k \to \mathbb{R}
```

for which you have values ``\ell(x)`` and the gradient ``\nabla \ell(x)``, it samples values from a density

```math
p(x) \propto \exp(\ell(x))
```

using the algorithm above.

[^MCMCintro]: For various techniques and a discussion of MCMC methods (eg domain transformations, or integrating out discrete parameters), you may find the [Stan Modeling Language manual](http://mc-stan.org/users/documentation/index.html) helpful. If you are unfamiliar with Bayesian methods, I would recommend [Bayesian Data Analysis](http://www.stat.columbia.edu/~gelman/book/) and [Statistical Rethinking](https://xcelab.net/rm/statistical-rethinking/).

The interface of DynamicHMC.jl expects that you code ``\ell(x), \nabla\ell(x)`` using the interface of the [LogDensityProblems.jl](https://github.com/tpapp/LogDensityProblems.jl) package, and also allows you to just code ``\ell(x)`` and obtain ``\nabla\ell(x)`` via [automatic differentiation](https://en.wikipedia.org/wiki/Automatic_differentiation).

While the NUTS algorithm operates on an *unrestricted* domain ``\mathbb{R}^k``, some parameters have natural restrictions: for example, standard deviations are positive, valid correlation matrices are a subset of all matrices, and structural econometric models can have parameter restrictions for stability. In order to sample for posteriors with parameters like these, *domain transformations* are required.[^Jacobian] Also, it is advantageous to decompose a flat vector `x` to a collection of parameters in a disciplined manner.

[^Jacobian]: For nonlinear transformations, correcting with the logarithm of the determinant of the Jacobian is required.

I recommend that you use [TransformVariables.jl](https://github.com/tpapp/TransformVariables.jl) in combination with LogdensityProblems.jl for this purpose: it has built-in transformations for common cases, and also allows decomposing vectors into tuples, named tuples, and arrays of parameters, combined with these transformations.

### Use cases

This package has the following intended use cases:

1. A robust and simple engine for MCMC. The intended audience is users who like to code their (log)posteriors directly, optimize and benchmark them them as Julia code, and at the same time want to have access detailed diagnostic information from the NUTS sampler.

2. A *backend* for another interface that needs a NUTS implementation.

3. A *research platform* for advances in MCMC methods. The code of this package is extensively documented, and should allow extensions and experiments easily using multiple dispatch. Contributions are always welcome.

### Support

If you have questions, feature requests, or bug reports, please [open an issue](https://github.com/tpapp/DynamicHMC.jl/issues/new). I would like to emphasize that it is perfectly fine to open issues just to ask questions. You can also address questions to [@Tamas_Papp](https://discourse.julialang.org/u/Tamas_Papp) on the Julia discourse forum.

## A worked example

!!! note
    An extended version of this example can be found [in the DynamicHMCExamples.jl package](https://github.com/tpapp/DynamicHMCExamples.jl/blob/master/src/example_independent_bernoulli.jl).

Consider estimating estimating the parameter ``0 \le \alpha \le 1`` from ``n`` IID observations ``y_i \sim \mathrm{Bernoulli}(\alpha)``.[^notedim] We will code this with the help of TransformVariables.jl, and obtain the gradient with ForwardDiff (in practice, for nontrivial models, at the moment I would recommend [Flux.jl](https://github.com/FluxML/Flux.jl)).

[^notedim]: Note that NUTS is not especially suitable for low-dimensional parameter spaces, but this example works fine.

First, we load the packages we use.

```@example bernoulli
using TransformVariables, LogDensityProblems, DynamicHMC,
    DynamicHMC.Diagnostics, Parameters, Statistics, Random
nothing # hide
```

Generally, I would recommend defining defining a structure to hold the data and all parameters relevant for the log density (eg the prior). This allows you to test your code in a modular way before sampling. For this model, the number of draws equal to `1` is a sufficient statistic.

```@example bernoulli
struct BernoulliProblem
    n::Int # Total number of draws in the data
    s::Int # Number of draws `==1` in the data
end
```

Then we make this problem *callable* with the parameters. Here, we have a single parameter `α`, but pass this in a `NamedTuple` to demonstrate a generally useful pattern. Then, we define an instance of this problem with the data, called `p`.

It is generally a good idea to test that your code works by calling it with the parameters; it should return a likelihood. For more complex models, you should benchmark and [optimize](https://docs.julialang.org/en/v1/manual/performance-tips/) this callable directly.

```@example bernoulli
function (problem::BernoulliProblem)(θ)
    @unpack α = θ               # extract the parameters
    @unpack n, s = problem       # extract the data
    # log likelihood, with constant terms dropped
    s * log(α) + (n-s) * log(1-α)
end

p = BernoulliProblem(20, 10)
p((α = 0.5, )) # make sure that it works
```

Note that here we used a *flat prior*. This is generally not a good idea for variables with non-finite support: one would usually make priors parameters of the `struct` above, and add the log prior to the log likelihood above.

With TransformVariables.jl, we set up a *transformation* ``\mathbb{R} \to [0,1]`` for ``\alpha``, and use the convenience function `TransformedLogDensity` to obtain a log density in ``\mathbb{R}^1``. Finally, we obtain a log density that supports gradients using automatic differentiation.

```@example bernoulli
trans = as((α = as𝕀,))
P = TransformedLogDensity(trans, p)
∇P = ADgradient(:ForwardDiff, P)
```

Finally, we run MCMC with warmup. Note that you have to specify the *random number generator* explicitly — this is good practice for parallel code. The last parameter is the number of samples.

```@example bernoulli
results = mcmc_with_warmup(Random.GLOBAL_RNG, ∇P, 1000; reporter = NoProgressReport())
nothing # hide
```

The returned parameter is a `NamedTuple`. Most importantly, it contains the field `chain`, which is a vector of vectors. You should use the transformation you defined above to retrieve the parameters (here, only `α`). We display the mean here to check that it was recovered correctly.

```@example bernoulli
posterior = transform.(Ref(trans), results.chain)
posterior_α = first.(posterior)
mean(posterior_α)
```

Using the `DynamicHMC.Diagnostics` submodule, you can obtain various useful diagnostics. The *tree statistics* in particular contain a lot of useful information about turning, divergence, acceptance rates, and tree depths for each step of the chain. Here we just obtain a summary.

```@example bernoulli
summarize_tree_statistics(results.tree_statistics)
```

!!! note
    Usually one would run parallel chains and check convergence and mixing using generic MCMC diagnostics not specific to NUTS. See [MCMCDiagnostics.jl](https://github.com/tpapp/MCMCDiagnostics.jl) for an implementation of ``\hat{R}`` and effective sample size calculations.

## Main user interface

The main entry point for sampling is

```@docs
mcmc_with_warmup
```

Warmup can be customized, and providing a fixed stepsize and kinetic energy is also possible.

```@docs
default_warmup_stages
fixed_stepsize_warmup_stages
InitialStepsizeSearch
FindLocalOptimum
DualAveraging
TuningNUTS
GaussianKineticEnergy
```

Progress reports can be explicit or silent.

```@docs
NoProgressReport
LogProgressReport
```

You probably won't need to change the tree building options with normal usage.

```@docs
DynamicHMC.TreeOptionsNUTS
```

## Diagnostics

!!! note
    The diagnostics API is not considered stable and may change without a deprecation cycle. It is intended for interactive use.

```@docs
DynamicHMC.Diagnostics.explore_log_acceptance_ratios
DynamicHMC.Diagnostics.summarize_tree_statistics
DynamicHMC.Diagnostics.leapfrog_trajectory
DynamicHMC.Diagnostics.EBFMI
DynamicHMC.PhasePoint
```
