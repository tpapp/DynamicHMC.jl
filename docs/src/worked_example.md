# A worked example

!!! note
    An extended version of this example can be found [in the DynamicHMCExamples.jl package](https://github.com/tpapp/DynamicHMCExamples.jl/blob/master/src/example_independent_bernoulli.jl).

## Problem statement

Consider estimating the parameter ``0 \le \alpha \le 1`` from ``n`` IID observations[^4]

[^4]: Note that NUTS is not especially suitable for low-dimensional parameter spaces, but this example works fine.

```math
y_i \sim \mathrm{Bernoulli}(\alpha)
```
We will code this with the help of TransformVariables.jl, and obtain the gradient with ForwardDiff.jl (in practice, at the moment I would recommend [ForwardDiff.jl](https://github.com/JuliaDiff/ForwardDiff.jl) for small models, and [Flux.jl](https://github.com/FluxML/Flux.jl) for larger ones — consider benchmarking a single evaluation of the log density with gradient).[^5]

[^5]: An example of how you can benchmark a log density with gradient `∇P`, obtained as described below:
    ```julia
    using BenchmarkTools, LogDensityProblems
    x = randn(LogDensityProblems.dimension(∇P))
    @benchmark LogDensityProblems.logdensity_and_gradient($∇P, $x)
    ```

## Coding up the log density

First, we load the packages we use.

```@example bernoulli
using TransformVariables, TransformedLogDensities, LogDensityProblems, LogDensityProblemsAD,
    DynamicHMC, DynamicHMC.Diagnostics, SimpleUnPack, Statistics, Random
nothing # hide
```

Generally, I would recommend defining an immutable composite type (ie `struct`) to hold the data and all parameters relevant for the log density (eg the prior). This allows you to test your code in a modular way before sampling. For this model, the number of draws equal to `1` is a sufficient statistic.

```@example bernoulli
struct BernoulliProblem
    n::Int # Total number of draws in the data
    s::Int # Number of draws `==1` in the data
end
```

Then we make this problem *callable* with the parameters. Here, we have a single parameter `α`, but pass this in a `NamedTuple` to demonstrate a generally useful pattern. Then, we define an instance of this problem with the data, called `p`.[^6]

[^6]: Note that here we used a *flat prior*. This is generally not a good idea for variables with non-finite support: one would usually make priors parameters of the `struct` above, and add the log prior to the log likelihood above.

```@example bernoulli
function (problem::BernoulliProblem)(θ)
    (; α) = θ               # extract the parameters
    (; n, s) = problem       # extract the data
    # log likelihood, with constant terms dropped
    s * log(α) + (n-s) * log(1-α)
end
```

It is generally a good idea to test that your code works by calling it with the parameters; it should return a likelihood. For more complex models, you should benchmark and [optimize](https://docs.julialang.org/en/v1/manual/performance-tips/) this callable directly.

```@example bernoulli
p = BernoulliProblem(20, 10)
p((α = 0.5, )) # make sure that it works
```

With [TransformVariables.jl](https://github.com/tpapp/TransformVariables.jl), we set up a *transformation* ``\mathbb{R} \to [0,1]`` for ``\alpha``, and use the convenience function `TransformedLogDensity` to obtain a log density in ``\mathbb{R}^1``. Finally, we obtain a log density that supports gradients using automatic differentiation, with [LogDensityProblemsAD.jl](https://github.com/tpapp/LogDensityProblemsAD.jl).

```@example bernoulli
trans = as((α = as𝕀,))
P = TransformedLogDensity(trans, p)
∇P = ADgradient(:ForwardDiff, P)
```

Finally, we run MCMC with warmup. Note that you have to specify the *random number generator* explicitly — this is good practice for parallel code. The last parameter is the number of samples.

```@example bernoulli
results = mcmc_with_warmup(Random.default_rng(), ∇P, 1000; reporter = NoProgressReport())
nothing # hide
```

The returned value is a `NamedTuple`. Most importantly, it contains the field `posterior_matrix`. You should use the transformation you defined above to retrieve the parameters (here, only `α`). We display the mean here to check that it was recovered correctly.

```@example bernoulli
posterior = transform.(trans, eachcol(results.posterior_matrix))
posterior_α = first.(posterior)
mean(posterior_α)
```

Using the [`DynamicHMC.Diagnostics`](@ref Diagnostics) submodule, you can obtain various useful diagnostics. The *tree statistics* in particular contain a lot of useful information about turning, divergence, acceptance rates, and tree depths for each step of the chain. Here we just obtain a summary.

```@example bernoulli
using DynamicHMC.Diagnostics
summarize_tree_statistics(results.tree_statistics)
```

## Parallel chains and diagnostics

Usually one would run multiple chains and check convergence and mixing using generic MCMC diagnostics not specific to NUTS.

The specifics of running multiple chains is up to the user: various forms of [parallel computing](https://docs.julialang.org/en/v1/manual/parallel-computing/) can be utilized depending on the problem scale and the hardware available. In the example below we use [multi-threading](https://docs.julialang.org/en/v1/manual/multi-threading/), using [ThreadTools.jl](https://github.com/baggepinnen/ThreadTools.jl); other excellent packages are available for threading.

It is easy to obtain posterior results for use with [MCMCDiagnosticsTools.jl](https://github.com/TuringLang/MCMCDiagnosticTools.jl/) with [`stack_posterior_matrices`](@ref):

```@example bernoulli
using ThreadTools, MCMCDiagnosticTools
results5 = tmap1(_ -> mcmc_with_warmup(Random.default_rng(), ∇P, 1000; reporter = NoProgressReport()), 1:5)
ess_rhat(stack_posterior_matrices(results5))
```

Use [`pool_posterior_matrices`](@ref) for a pooled sample:

```@example bernoulli
posterior5 = transform.(trans, eachcol(pool_posterior_matrices(results5)))
posterior5_α = first.(posterior5)
mean(posterior5_α)
```
