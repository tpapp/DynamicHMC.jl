using DynamicHMC
using Distributions
using PDMats
using Base.Test
using Parameters
using ArgCheck

import DynamicHMC: logdensity, loggradient
import ForwardDiff: gradient

# consistent testing
const RNG = srand(UInt32[0x23ef614d, 0x8332e05c, 0x3c574111, 0x121aa2f4])

import DynamicHMC: GaussianKE, Hamiltonian, loggradient, logdensity,
    phasepoint, rand_phasepoint, leapfrog, move


"Random positive definite matrix of size `n` x `n` (for testing)."
function rand_PDMat(n)
    A = randn(n,n)
    PDMat(A'*A)
end

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

"Random Hamiltonian H with phasepoint z."
function rand_Hz(N)
    μ = randn(N)
    Σ = rand_PDMat(N)
    κ = GaussianKE(PDiagMat(1./abs.(randn(N))))
    H = Hamiltonian(MvNormal(μ, Σ), κ)
    z = rand_phasepoint(RNG, H, μ)
    H, z
end
