import DynamicHMC: logdensity, loggradient, GaussianKE

import Base: rand

"""
Normal density, used for testing. Reuses code from Gaussian kinetic energy.
"""
struct NormalDensity{T,S <: GaussianKE}
    "Mean."
    μ::Vector{T}
    κ::S
end

"Calculate `U` that makes `U'*U*Σ = I`."
ΣtoU(Σ) = chol(inv(Symmetric(Σ)))

normal_density(μ, Σ) = NormalDensity(μ, GaussianKE(ΣtoU(Σ)))
function normal_density(μ, Σ::Diagonal)
    NormalDensity(μ, GaussianKE(Diagonal(1./.√diag(Σ))))
end
function normal_density(μ, Σ::UniformScaling)
    NormalDensity(μ, GaussianKE(Diagonal(fill(1/√Σ.λ, length(μ)))))
end
logdensity(ℓ::NormalDensity, p) = logdensity(ℓ.κ, p-ℓ.μ)
loggradient(ℓ::NormalDensity, p) = loggradient(ℓ.κ, p-ℓ.μ)
Base.rand(ℓ::NormalDensity) = rand(ℓ.κ) + ℓ.μ

@testset "normal density" begin
    μ = [1.0, 2.0, -1.0]
    A = [1.0 0.1 0;
         0.1 2.0 0.3
         0 0.3 3.0]
    Σ = A'*A
    d = normal_density(μ, Σ)
    M = 1000000
    z = Array{Float64}(length(μ), M)
    for i in 1:M
        z[:, i] = rand(d)
    end
    @test norm(mean(z, 2)-μ, Inf) ≤ 0.01
    @test norm(cov(z, 2)-Σ, Inf) ≤ 0.03
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

"Random positive definite matrix of size `n` x `n` (for testing)."
function rand_PDmat(n)
    A = randn(n,n)
    A'*A
end
