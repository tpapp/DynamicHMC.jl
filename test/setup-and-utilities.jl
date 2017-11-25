using DynamicHMC

import DynamicHMC:
    Hamiltonian, GaussianKE, PhasePoint, rand_phasepoint, loggradient

using Base.Test

using ArgCheck
using DiffResults
using Distributions
import ForwardDiff: gradient
using MCMCDiagnostics
using Parameters
using StatsBase

"RNG for consistent test environment"
const RNG = srand(UInt32[0x23ef614d, 0x8332e05c, 0x3c574111, 0x121aa2f4])

"Be more tolerant when testing."
const RELAX = (k = "CONTINUOUS_INTEGRATION"; haskey(ENV, k) && ENV[k] == "true")

"Random positive definite matrix of size `n` x `n` (for testing)."
function rand_Σ(::Type{Symmetric}, n)
    A = randn(RNG, n,n)
    Symmetric(A'*A+0.01)
end

rand_Σ(::Type{Diagonal}, n) = Diagonal(randn(RNG, n).^2+0.01)

rand_Σ(n::Int) = rand_Σ(Symmetric, n)

"Test `loggradient` vs autodiff `neg_energy`."
function test_loggradient(ℓ, x)
    ∇ = loggradient(ℓ, x)
    ∇_AD = ForwardDiff.gradient(x->neg_energy(ℓ,x), x)
    @test ∇ ≈ ∇_AD
end

## use MvNormal as a test distribution
(ℓ::MvNormal)(p) = DiffResults.DiffResult(logpdf(ℓ, p), (gradlogpdf(ℓ, p), ))

"Lenient comparison operator for `struct`, both mutable and immutable."
@generated function ≂(x, y)
    if !isempty(fieldnames(x)) && x == y
        mapreduce(n -> :(x.$n == y.$n), (a,b)->:($a && $b), fieldnames(x))
    else
        :(x == y)
    end
end

"Random Hamiltonian `H` with phasepoint `z`, with dimension `K`."
function rand_Hz(K)
    μ = randn(K)
    Σ = rand_Σ(K)
    κ = GaussianKE(inv(rand_Σ(Diagonal, K)))
    H = Hamiltonian(MvNormal(μ, full(Σ)), κ)
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
    m, C = mean_and_cov(x)
    vec(mean(x, 1)), cov(x, 1)
end

"""
    find_stable_ϵ(κ, Σ)

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
find_stable_ϵ(κ::GaussianKE, Σ) = √eigmin(κ.W'*Σ*κ.W)

function find_stable_ϵ(H::Hamiltonian{Tℓ, Tκ}) where
    {Tℓ <: Distribution{Multivariate,Continuous}, Tκ}
    find_stable_ϵ(H.κ, cov(H.ℓ))
end

"Simple Hamiltonian Monte Carlo transition, for testing."
function simple_HMC(rng, H, z::PhasePoint, ϵ, L)
    π₀ = neg_energy(H, z)
    z′ = z
    for _ in 1:L
        z′ = leapfrog(H, z′, ϵ)
    end
    Δ = neg_energy(H, z′) - π₀
    accept = Δ > 0 || (rand(rng) < exp(Δ))
    accept ? z′ : z
end

"""
    sample_HMC(rng, H, q, N; ϵ = find_stable_ϵ(H), L = 10)

Simple Hamiltonian Monte Carlo sample, for testing.
"""
function sample_HMC(rng, H, q, N; ϵ = find_stable_ϵ(H), L = 10)
    qs = similar(q, N, length(q))
    for i in 1:N
        z = rand_phasepoint(RNG, H, q)
        q = simple_HMC(RNG, H, z, ϵ, L).q
        qs[i, :] = q
    end
    qs
end
