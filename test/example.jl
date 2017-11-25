# install unregistered packages
# Pkg.clone("https://github.com/tpapp/ContinuousTransformations.jl")
# Pkg.clone("https://github.com/tpapp/DiffWrappers.jl")
# Pkg.clone("https://github.com/tpapp/MCMCDiagnostics.jl")
using ContinuousTransformations
using DiffWrappers
using DynamicHMC
using MCMCDiagnostics
using Parameters

"""
Toy problem using a Bernoulli distribution.

We model `n` independent draws from a ``Bernoulli(α)`` distribution.
"""
struct BernoulliProblem
    "Total number of draws in the data."
    n::Int
    "Number of draws =1 in the data."
    s::Int
end

function (problem::BernoulliProblem)(α)
    @unpack n, s = problem
    s * log(α) + (n-s) * log(1-α) # log likelihood
end

p = BernoulliProblem(100, 40)                             # original problem
pt = TransformLogLikelihood(p, bridge(ℝ, Segment(0, 1)))  # transform
pt∇ = ForwardGradientWrapper(pt, [0.0]);                  # AD using ForwardDiff.jl

sample, NUTS_tuned = NUTS_init_tune_mcmc(Base.Random.GLOBAL_RNG, pt∇, 1, 1000);
posterior = first.(map(get_transformation(pt) ∘ get_position, sample));

effective_sample_size(posterior)
NUTS_statistics(sample)         # NUTS-specific statistics
