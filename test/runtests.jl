using DynamicHMC

using DynamicHMC:
    # Hamiltonian
    GaussianKE, Hamiltonian, PhasePoint, neg_energy, phasepoint_in,
    rand_phasepoint, leapfrog, move, is_valid_ℓq,
    # building blocks
    rand_bool, TurnStatistic, combine_turnstats, Proposal,
    combined_logprob_logweight, combine_proposals,
    DivergenceStatistic, combine_divstats, divergence_statistic, isdivergent,
    get_acceptance_rate, isturning, adjacent_tree, sample_trajectory,
    # stepsize
    InitialStepsizeSearch, find_initial_stepsize,
    # transitions and tuning
    NUTS_transition, NUTS_init, StepsizeTuner, StepsizeCovTuner, tune

using Test

using ArgCheck: @argcheck
using DataStructures
import DiffResults
using DiffResults: MutableDiffResult
using Distributions
using DocStringExtensions: SIGNATURES
import ForwardDiff
using LinearAlgebra
using MCMCDiagnostics: effective_sample_size, potential_scale_reduction
using Parameters
import Random
using Random: randn, rand
using StatsBase: mean_and_cov, mean_and_std
using StatsFuns: logaddexp
using Statistics: mean, quantile
using Suppressor


# general test environment

const RNG = Random.GLOBAL_RNG   # shorthand
Random.seed!(RNG, UInt32[0x23ef614d, 0x8332e05c, 0x3c574111, 0x121aa2f4])

"Tolerant testing in a CI environment."
const RELAX = (k = "CONTINUOUS_INTEGRATION"; haskey(ENV, k) && ENV[k] == "true")

"""
    $SIGNATURES

Random positive definite matrix of size `n` x `n` (for testing).
"""
function rand_Σ(::Type{Symmetric}, n)
    A = randn(n, n)
    Symmetric(A'*A .+ 0.01)
end

rand_Σ(::Type{Diagonal}, n) = Diagonal(randn(n).^2 .+ 0.01)

rand_Σ(n::Int) = rand_Σ(Symmetric, n)

## use MvNormal as a test distribution
(ℓ::MvNormal)(p) = DiffResults.DiffResult(logpdf(ℓ, p), (gradlogpdf(ℓ, p), ))

"Random Hamiltonian `H` with phasepoint `z`, with dimension `K`."
function rand_Hz(K)
    μ = randn(K)
    Σ = rand_Σ(K)
    κ = GaussianKE(inv(rand_Σ(Diagonal, K)))
    H = Hamiltonian(MvNormal(μ, Matrix(Σ)), κ)
    z = rand_phasepoint(RNG, H, μ)
    H, z
end

"""
    simulated_meancov(f, N)

Simulated mean and covariance of `N` values from `f()`.
"""
function simulated_meancov(f, N)
    s = f()
    K = length(s)
    x = similar(s, (N, K))
    for i in 1:N
        x[i, :] = f()
    end
    m, C = mean_and_cov(x, 1)
    vec(m), C
end

include("test-utilities.jl")
include("test-Hamiltonian-leapfrog.jl")
include("test-buildingblocks.jl")
include("test-stepsize.jl")
include("test-sample-dummy.jl")
include("test-tuners.jl")
include("test-sample-normal.jl")
include("test-normal-mcmc.jl")
include("test-statistics.jl")
include("test-reporting.jl")

include("../docs/make.jl")
