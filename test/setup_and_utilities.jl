using DynamicHMC

using DynamicHMC:
    Hamiltonian, GaussianKE, PhasePoint, rand_phasepoint, loggradient,
    neg_energy, leapfrog

using Test
using Random: srand, AbstractRNG
import Random: rand
using LinearAlgebra

using ArgCheck: @argcheck
using DataStructures: counter
import DiffResults
# using Distributions
import ForwardDiff
# using MCMCDiagnostics
using Parameters: @unpack
using DocStringExtensions: SIGNATURES
using StatsFuns: chisqinvcdf
import StatsBase
using StatsBase: mean_and_cov, mean_and_std, std, var
using SymmetricProducts: symprod
using Suppressor: @capture_err, @color_output

"RNG for consistent test environment"
const RNG = srand(UInt32[0x23ef614d, 0x8332e05c, 0x3c574111, 0x121aa2f4])

"Be more tolerant when testing."
const RELAX = (k = "CONTINUOUS_INTEGRATION"; haskey(ENV, k) && ENV[k] == "true")


# general test utilities

"""
    $SIGNATURES

Simulated mean and covariance of `N` values from `f()`.
"""
function simulated_meancov(f, N)
    s = f()
    x = similar(s, (N, length(s)))
    for i in 1:N
        x[i, :] = f()
    end
    m, C = mean_and_cov(x, 1)
    vec(m), C
end

"""
    $SIGNATURES

A relative difference measure, primarily for comparing `expected` and
`simulated` values, normalizing by the former (adjusted to 1 when small).

Uses the `p`-norm (defaulting to the maximum norm).
"""
function reldiff(expected, simulated, p = Inf)
    rawnorm = norm((@. (simulated - expected) / max(1, abs(expected))), Inf)
    if p == Inf
        rawnorm
    else
        rawnorm / (length(expected)^(1/p))
    end
end

"""
    $SIGNATURES

Test `loggradient` vs autodiff `neg_energy`.
"""
function test_loggradient(ℓ, x)
    ∇ = loggradient(ℓ, x)
    ∇_AD = ForwardDiff.gradient(x->neg_energy(ℓ, x), x)
    @test ∇ ≈ ∇_AD
end

"""
    $SIGNATURES

Lenient comparison operator for `struct`, both mutable and immutable.
"""
@generated function ≂(x, y)
    if !isempty(fieldnames(x)) && x == y
        mapreduce(n -> :(x.$n == y.$n), (a,b)->:($a && $b), fieldnames(x))
    else
        :(x == y)
    end
end

## for testing ≂
struct Foo{T}
    a::T
    b::T
end

@testset "≂ comparisons" begin
    @test 1 ≂ 1
    @test 1 ≂ 1.0
    @test [1,2] ≂ [1,2]
    @test [1.0,2.0] ≂ [1,2]
    @test !(1 ≂ 2)
    @test Foo(1,2) ≂ Foo(1,2)
    @test !(Foo(1,2) ≂ Foo(1,3))
    @test !(Foo{Any}(1,2) ≂ Foo(1,2))
end


# random covariance matrix

"""
    $SIGNATURES

Random positive definite matrix of size `n` x `n` (for testing).
"""
function rand_Σ(rng::AbstractRNG, ::Type{Symmetric}, n)
    A = randn(rng, n,n)
    Symmetric(A' * A .+ 0.01)
end

rand_Σ(rng::AbstractRNG, ::Type{Diagonal}, n) = Diagonal(randn(rng, n).^2 .+ 0.01)

rand_Σ(rng::AbstractRNG, n::Int) = rand_Σ(rng, Symmetric, n)




"""
    $SIGNATURES

Convert `Σ` (variance) to `U` (Cholesky factor of ``Σ⁻¹=U'U``).
"""
Σ2U(Σ::AbstractMatrix) = cholesky(inv(Σ)).U

# TODO: revisit and remove workaround for
# https://github.com/JuliaLang/julia/issues/27821
Σ2U(D::Diagonal) = .√inv(D)

"""
A multivariate normal distribution for testing. Uses the Cholesky factor of
precision (inverse variance).
"""
struct MultiNormal{Tμ <: AbstractVector, TU <: AbstractMatrix}
    "mean"
    μ::Tμ
    "``U'U=(Σ)⁻¹"
    U::TU
    function MultiNormal((μ, U)::NamedTuple{(:μ,:U),Tuple{Tμ, TU}}) where {Tμ,TU}
        @argcheck length(μ) == LinearAlgebra.checksquare(U)
        new{Tμ,TU}(μ, U)
    end
end

function MultiNormal(μ, Σ)
    U = Σ2U(Σ)
    MultiNormal((μ = μ, U = U))
end

function logpdf(m::MultiNormal, x)
    @unpack μ, U = m
    n = length(x)
    z = U * (x - μ)
    -(log(2*π)*n/2 + logdet(U) + sum(abs2, z)/2)
end

∇logpdf(m::MultiNormal, x) = - m.U' * (m.U * (x - m.μ))

StatsBase.var(m::MultiNormal) = inv(symprod(m.U'))

Base.mean(m::MultiNormal) = m.μ

Base.length(m::MultiNormal) = length(m.μ)

function rand(rng::AbstractRNG, m::MultiNormal)
    @unpack μ, U = m
    μ + (U \ randn(rng, length(μ)))
end

(ℓ::MultiNormal)(p) = DiffResults.DiffResult(logpdf(ℓ, p), (∇logpdf(ℓ, p), ))



"Random Hamiltonian `H` with phasepoint `z`, with dimension `K`."
function rand_Hz(rng, K)
    μ = randn(rng, K)
    Σ = rand_Σ(rng, K)
    κ = GaussianKE(inv(rand_Σ(rng, Diagonal, K)))
    H = Hamiltonian(MultiNormal(μ, Σ), κ)
    z = rand_phasepoint(rng, H, μ)
    H, z
end

@testset "multivariate normal distribution" begin
    Σ = Symmetric([2.16612 1.71727 3.12289;
                   1.71727 3.37755 0.940824;
                   3.12289 0.940824 6.90186])
    μ = [0.436657, 0.184227, 0.585647]
    d = MultiNormal(μ, Σ)
    @test var(d) ≈ Σ
    m, C = simulated_meancov(()->rand(RNG, d), 10000)
    @test reldiff(μ, m) ≤ 0.05
    @test reldiff(Σ, C) ≤ 0.1
end

"""
    $SIGNATURES

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

# TODO revisit and remove workaround for
# https://github.com/JuliaLang/julia/issues/27847
LinearAlgebra.eigmin(d::Diagonal) = minimum(diag(d))

find_stable_ϵ(H::Hamiltonian{<: MultiNormal}) = find_stable_ϵ(H.κ, var(H.ℓ))

"""
    $SIGNATURES

Simple Hamiltonian Monte Carlo transition, for testing.
"""
function HMC_transition(rng, H, z::PhasePoint, ϵ, L)
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
    $SIGNATURES

Simple Hamiltonian Monte Carlo sample, for testing.
"""
function sample_HMC(rng, H, q, N; ϵ = find_stable_ϵ(H), L = 10)
    qs = similar(q, N, length(q))
    for i in 1:N
        z = rand_phasepoint(rng, H, q)
        q = HMC_transition(rng, H, z, ϵ, L).q
        qs[i, :] = q
    end
    qs
end
