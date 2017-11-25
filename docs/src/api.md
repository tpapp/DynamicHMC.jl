```@meta
CurrentModule = DynamicHMC
```

# [The density function ``\ell``: an example](@id ell-tutorial)

The density function should take a single argument `q`, which is a vector of numbers, and return an object which provides the methods `DiffResults.value` and `DiffResults.gradient` to access ``\ell(q)`` and ``\nabla\ell(q)``, respectively.

The following example implements the density function for ``n`` observations from a ``\text{Bernoulli}(\alpha)`` distribution, ``s`` of which are `1`. The complete example is available in [`tests/example.jl`](https://github.com/tpapp/DynamicHMC.jl/blob/master/test/test-sample-dummy.jl).

It is convenient to define a structure that holds the data,

```julia
struct BernoulliProblem
    "Total number of draws in the data."
    n::Int
    "Number of draws =1 in the data."
    s::Int
end
```

then make it callable:

```julia
function (problem::BernoulliProblem)(α)
    @unpack n, s = problem        # using Parameters
    s * log(α) + (n-s) * log(1-α) # log likelihood
end
```

and finally define an object with actual data:

```julia
p = BernoulliProblem(100, 40)                             # original problem
```

The value `p` is a function that takes a single real number, and returns the likelihood. However, the functions in this package

1. take a **vector** which contains elements in ``\mathbb{R}``, and

2. expect ``\ell`` (ie `p` above) to return an object that can provide the **value and the derivatives**.

We could implement both manually, but it is convenient to use wrappers from two packages mentioned in the [introduction](@ref Introduction):

```julia
pt = TransformLogLikelihood(p, bridge(ℝ, Segment(0, 1)))  # transform
pt∇ = ForwardGradientWrapper(pt, [0.0]);                  # AD using ForwardDiff.jl
```

Then we can call the high-level function [`NUTS_init_tune_mcmc`](@ref) that initializes and tunes the sampler, and samples from it:

```julia
sample, NUTS_tuned = NUTS_init_tune_mcmc(pt∇, 1, 1000);
```

The returned objects are the *sample*, which contains the draws and diagnostic information, and the tuned sampler, which we could use to continue sampling.

We obtain the posterior using the transformation and [`get_position`](@ref):

```julia
posterior = map(get_transformation(pt) ∘ get_position, sample);
```

which is a vector of vectors. Calculate the effective sample size and NUTS-specific statistics as

```julia
julia> effective_sample_size(first.(posterior))
323.6134099739428

julia> NUTS_statistics(sample)         # NUTS-specific statistics
Hamiltonian Monte Carlo sample of length 1000
  acceptance rate mean: 0.92, min/25%/median/75%/max: 0.26 0.87 0.97 1.0 1.0
  termination: AdjacentTurn => 31% DoubledTurn => 69%
  depth: 1 => 66% 2 => 34%
```

# Sampling and accessors

Most users would use this function, which initializes and tunes the parameters of the algorithm, then samples. Parameters can be set manually for difficult posteriors.

```@docs
NUTS_init_tune_mcmc
```

These functions can be used use to perform the steps above manually.

```@docs
NUTS_init
tune
mcmc
```

The resulting sample is a vector of [`NUTS_Transition`](@ref) objects, for which the following accessors exist:

```@docs
NUTS_Transition
get_position
get_neg_energy
get_depth
get_termination
get_acceptance_rate
get_steps
get_position_matrix
```

!!! important

    The [`NUTS`](@ref) sampler saves a random number generator and uses it for random draws. When running in parallel, you should initialize [`NUTS_init_tune_mcmc`](@ref) with a random number generator as its first argument explicitly, making sure that each thread has its own one.

# Diagnostics

These are NUTS-specific diagnostics and statistics (except for [`sample_cov`](@ref), which is a convenience function). It is also prudent to use generic MCMC convergence diagnostics, as suggested in the [introduction](@ref Introduction).

```@docs
NUTS_statistics
sample_cov
EBFMI
```

# Fine-grained control

## Kinetic energy

```@docs
KineticEnergy
EuclideanKE
GaussianKE
```

## [NUTS parameters and tuning](@id tuning)

```@docs
NUTS
AbstractTuner
StepsizeTuner
StepsizeCovTuner
TunerSequence
mcmc_adapting_ϵ
bracketed_doubling_tuner
```
