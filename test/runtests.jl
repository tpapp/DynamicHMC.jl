using DynamicHMC

import DynamicHMC: logdensity, loggradient
using Base.Test

"Normal density, used for testing."
struct NormalDensity{T,S}
    "Mean."
    μ::Vector{T}
    "Square root of the *inverse* of the variance matrix Σ, ie L'L=Σ."
    L::S
end

logdensity(ℓ::NormalDensity, p) = -sum(abs2, ℓ.L*(p-ℓ.μ))/2
loggradient(ℓ::NormalDensity, p) = -ℓ.L'*ℓ.L*(p-ℓ.μ)
Base.rand(ℓ::NormalDensity) = ℓ.L \ (randn(length(ℓ.μ)) + ℓ.μ)

"""
Test that the Hamiltonian is invariant using the leapfrog integrator,
normal with identity matrix.
"""
@testset "leapfrog" begin
    ω = UNITNORMAL
    ℓ = NormalDensity(fill(0.0, 3), I)
    q = rand(ℓ)
    p = rand(ω, q)
    minuslogH₀ = minuslogH(ℓ, ω, q, p)
    for i in 1:50
        q, p = leapfrog(ℓ, ω, q, p, 0.01)
        @test isapprox(minuslogH₀, minuslogH(ℓ, ω, q, p); rtol = 1e-3)
    end
end

"Test that the Hamiltonian is invariant using the leapfrog integrator,
non-centered normal with correlated matrix."
@testset "leapfrog" begin
    ω = UNITNORMAL
    ℓ = NormalDensity(fill(0.0,3), [1.0 0.1 0.2;
                                   0 2.0 0.3
                                   0 0 0.5])
    q = rand(ℓ)
    p = rand(ω, q)
    minuslogH₀ = minuslogH(ℓ, ω, q, p)
    for i in 1:500
        q, p = leapfrog(ℓ, ω, q, p, 0.1)
        @test isapprox(minuslogH₀, minuslogH(ℓ, ω, q, p); rtol = 1e-3)
    end
end
