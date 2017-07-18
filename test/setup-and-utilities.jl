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

@testset "MvNormal loggradient" begin
    for _ in 1:10
        n = rand(2:6)
        ℓ = MvNormal(randn(n), rand_PDMat(n))
        for _ in 1:10
            test_loggradient(ℓ, randn(n))
        end
    end
end

"Lenient comparison operator for `struct`, both mutable and immutable."
@generated function ≂(x, y)
    if !isempty(fieldnames(x)) && x == y
        mapreduce(n -> :(x.$n == y.$n), (a,b)->:($a && $b), fieldnames(x))
    else
        :(x == y)
    end
end

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

"Random Hamiltonian H with phasepoint z."
function rand_Hz(N)
    μ = randn(N)
    Σ = rand_PDMat(N)
    κ = GaussianKE(PDiagMat(1./abs.(randn(N))))
    H = Hamiltonian(MvNormal(μ, Σ), κ)
    z = rand_phasepoint(RNG, H, μ)
    H, z
end
