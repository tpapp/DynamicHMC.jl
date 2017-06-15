using DynamicHMC

import DynamicHMC: logdensity, loggradient
using Base.Test

# consistent testing
srand(UInt32[0x23ef614d, 0x8332e05c, 0x3c574111, 0x121aa2f4])

"""
Normal density, used for testing. Reuses code from Gaussian kinetic energy.
"""
struct NormalDensity{T,S <: GaussianKE}
    "Mean."
    μ::Vector{T}
    κ::S
end

normal_density(μ, Σ) = NormalDensity(μ, GaussianKE(chol(inv(Symmetric(Σ)))))
function normal_density(μ, Σ::Diagonal)
    NormalDensity(μ, GaussianKE(Diagonal(1./.√diag(Σ))))
end
function normal_density(μ, Σ::UniformScaling)
    NormalDensity(μ, GaussianKE(Diagonal(fill(1/√Σ.λ, length(μ)))))
end
logdensity(ℓ::NormalDensity, p) = logdensity(ℓ.κ, p-ℓ.μ)
loggradient(ℓ::NormalDensity, p) = loggradient(ℓ.κ, p-ℓ.μ)
Base.rand(ℓ::NormalDensity) = propose(ℓ.κ) + ℓ.μ

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

"""
Test that the Hamiltonian is invariant using the leapfrog integrator.
"""
function test_hamiltonian_invariance(hmc, N; atol=1e-3)
    q = rand(hmc.ℓ)
    p = propose(hmc.κ, q)
    π₀ = logdensity(hmc, q, p)
    for _ in 1:N
        q, p = leapfrog(hmc, q, p)
        @test isapprox(π₀, logdensity(hmc, q, p); atol = atol)
    end
end

@testset "leapfrog" begin
    test_hamiltonian_invariance(HMC(normal_density(fill(0.0, 3), I), 
                                    GaussianKE(Diagonal(ones(3))),
                                    0.01), 100)
    A = rand(3,3)
    test_hamiltonian_invariance(HMC(normal_density(randn(3), A'*A), 
                                    GaussianKE(Diagonal([1.0,0.5,2.0])),
                                    0.001), 100)
end
