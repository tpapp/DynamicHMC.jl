#####
##### include this separately for interactive tests
#####

####
#### packages and symbols
####

using DynamicHMC, Test, ArgCheck, DocStringExtensions, HypothesisTests, LinearAlgebra,
    MCMCDiagnostics, Parameters, Random, StatsBase, StatsFuns, Statistics

import ForwardDiff, Random, TransformVariables

using DynamicHMC:
    # trees
    Directions, next_direction, biased_progressive_logprob2, adjacent_tree,
    sample_trajectory, InvalidTree,
    # Hamiltonian
    GaussianKineticEnergy, kinetic_energy, ∇kinetic_energy, rand_p, Hamiltonian,
    EvaluatedLogDensity, evaluate_ℓ, PhasePoint, logdensity, leapfrog, calculate_p♯,
    logdensity,
    # NUTS
    TrajectoryNUTS, rand_bool, GeneralizedTurnStatistic, AcceptanceStatistic,
    leaf_acceptance_statistic, acceptance_rate, TreeStatisticsNUTS, NUTS, sample_tree,
    # stepsize
    find_crossing_stepsize, bisect_stepsize, find_initial_stepsize, InitialStepsizeSearch,
    DualAveraging, initial_adaptation_state, adapt_stepsize, current_ϵ, final_ϵ,
    FixedStepsize,
    # mcmc
    position_matrix, WarmupState

import DynamicHMC:
    # trees
    move, is_turning, combine_turn_statistics, is_divergent,
    combine_visited_statistics, calculate_logprob2, combine_proposals, leaf

using DynamicHMC.Diagnostics
using DynamicHMC.Diagnostics: ACCEPTANCE_QUANTILES

using LogDensityProblems: logdensity_and_gradient, dimension, LogDensityProblems

### uncomment code below to use latest LogDensityTestSuite
# if !isinteractive()             # on CI
#     @info "installing LogDensityTestSuite#master"
#     import Pkg
#     Pkg.API.add(Pkg.PackageSpec(; name = "LogDensityTestSuite", rev = "master"))
# end
using LogDensityTestSuite

####
#### utilities for testing
####

###
### general test environment
###

const RNG = Random.GLOBAL_RNG   # shorthand
Random.seed!(RNG, UInt32[0x23ef614d, 0x8332e05c, 0x3c574111, 0x121aa2f4])

"Tolerant testing in a CI environment."
const RELAX = (k = "CONTINUOUS_INTEGRATION"; haskey(ENV, k) && ENV[k] == "true")

###
### random values
###

"""
$(SIGNATURES)

Random positive definite matrix of size `n` x `n` (for testing).
"""
function rand_Σ(::Type{Symmetric}, n)
    A = randn(n, n)
    Symmetric(A'*A .+ 0.01)
end

rand_Σ(::Type{Diagonal}, n) = Diagonal(randn(n).^2 .+ 0.01)

rand_Σ(n::Int) = rand_Σ(Symmetric, n)

"""
$(SIGNATURES)

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

@testset "simulated meancov" begin
    μ = [2, 1.2]
    D = [2.0, 0.7]
    m, C = simulated_meancov(()-> randn(2) .* D .+ μ, 10000)
    @test m ≈ μ atol = 0.05 rtol = 0.1
    @test C ≈ Diagonal(abs2.(D)) atol = 0.05 rtol = 0.1
end

###
### Multivariate normal ℓ for testing
###

"Random (positive) diagonal matrix."
rand_D(K) = Diagonal(abs.(randn(K)))

"Random Cholesky factor for correlation matrix."
function rand_C(K)
    t = TransformVariables.CorrCholeskyFactor(K)
    t(randn(TransformVariables.dimension(t)))
end

"Multivariate normal with `Σ = LL'`."
multivariate_normal(μ, L) = shift(μ, linear(L, StandardMultivariateNormal(length(μ))))

"Multivariate normal with diagonal `Σ` (constant `v` variance)."
multivariate_normal(μ, v::Real = 1) = multivariate_normal(μ, Diagonal(fill(v, length(μ))))

###
### Hamiltonian test helper functions
###

"""
$(SIGNATURES)

Return a reasonable estimate for the largest stable stepsize (which may not be
stable, but is a good starting point for finding that).

`q` is assumed to be normal with variance `Σ`. `κ` is the kinetic energy.

Using the transformation ``p̃ = W⁻¹ p``, the kinetic energy is

``p'M⁻¹p = p'W⁻¹'W⁻¹p/2=p̃'p̃/2``

Transforming to ``q̃=W'q``, the variance of which becomes ``W' Σ W``. Return the
square root of its smallest eigenvalue, following Neal (2011, p 136).

When ``Σ⁻¹=M=WW'``, this the variance of `q̃` is ``W' Σ W=W' W'⁻¹W⁻¹W=I``, and
thus decorrelates the density perfectly.
"""
find_stable_ϵ(κ::GaussianKineticEnergy, Σ) = √eigmin(κ.W'*Σ*κ.W)

"""
$(SIGNATURES)

A `NamedTuple` that contains

- a random `K`-element vector `μ`

- a random `K × K` covariance matrix `Σ`,

- a random Hamiltonian `H` with `ℓ` corresponding to a multivariate normal with `μ`, `Σ`,
  and a random Gaussian kinetic energy (unrelated to `ℓ`)

- a random phasepoint `z`.

Useful for testing.
"""
function rand_Hz(K)
    μ = randn(K)
    Σ = rand_Σ(K)
    L = cholesky(Σ).L
    κ = GaussianKineticEnergy(inv(rand_Σ(Diagonal, K)))
    H = Hamiltonian(κ, multivariate_normal(μ, L))
    z = PhasePoint(evaluate_ℓ(H.ℓ, μ .+ L * randn(K)), rand_p(RNG, κ))
    (μ = μ, Σ = Σ, H = H, z = z)
end
