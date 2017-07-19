using DynamicHMC
using Distributions
using Base.Test
using Parameters
using ArgCheck

import DynamicHMC: logdensity, loggradient
import ForwardDiff: gradient

import DynamicHMC: GaussianKE, Hamiltonian, loggradient, logdensity,
    phasepoint, rand_phasepoint, leapfrog, move

"RNG for consistent test environment"
const RNG = srand(UInt32[0x23ef614d, 0x8332e05c, 0x3c574111, 0x121aa2f4])

"Random positive definite matrix of size `n` x `n` (for testing)."
function rand_Σ(::Type{Symmetric}, n)
    A = randn(RNG, n,n)
    Symmetric(A'*A+0.01)
end

rand_Σ(::Type{Diagonal}, n) = Diagonal(randn(RNG, n).^2+0.01)

rand_Σ(n::Int) = rand_Σ(Symmetric, n)

"Test `loggradient` vs autodiff `logdensity`."
function test_loggradient(ℓ, x)
    ∇ = loggradient(ℓ, x)
    ∇_AD = ForwardDiff.gradient(x->logdensity(ℓ,x), x)
    @test ∇ ≈ ∇_AD
end

## use MvNormal as a test distribution
logdensity(ℓ::MvNormal, p) = logpdf(ℓ, p)
loggradient(ℓ::MvNormal, p) = -(ℓ.Σ \ (p - ℓ.μ))

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
    vec(mean(x, 1)), cov(x, 1)
end
