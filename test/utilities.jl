#####
##### utilities for testing
#####

####
#### general test environment
####

const RNG = Random.GLOBAL_RNG   # shorthand
Random.seed!(RNG, UInt32[0x23ef614d, 0x8332e05c, 0x3c574111, 0x121aa2f4])

"Tolerant testing in a CI environment."
const RELAX = (k = "CONTINUOUS_INTEGRATION"; haskey(ENV, k) && ENV[k] == "true")

####
#### random values
####

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
    d = MvNormal([2,2], [2.0,1.0])
    m, C = simulated_meancov(()->rand(d), 10000)
    @test m ≈ mean(d) atol = 0.05 rtol = 0.1
    @test C ≈ cov(d) atol = 0.05 rtol = 0.1
end


####
#### use multivariate distributions as tests
####

"""
Obtain the log density from a distribution.
"""
struct DistributionLogDensity{D <: Distribution{Multivariate,Continuous}}
    distribution::D
end

dimension(ℓ::DistributionLogDensity) = length(ℓ.distribution)

capabilities(::Type{<:DistributionLogDensity}) = LogDensityProblems.LogDensityOrder(1)

function logdensity_and_gradient(ℓ::DistributionLogDensity, x::AbstractVector)
    logpdf(ℓ.distribution, x), gradlogpdf(ℓ.distribution, x)
end

DistributionLogDensity(::Type{MvNormal}, n::Int) = # canonical
    DistributionLogDensity(MvNormal(zeros(n), ones(n)))

Statistics.mean(ℓ::DistributionLogDensity) = mean(ℓ.distribution)
Statistics.var(ℓ::DistributionLogDensity) = var(ℓ.distribution)
Statistics.cov(ℓ::DistributionLogDensity) = cov(ℓ.distribution)
Base.rand(ℓ::DistributionLogDensity) = rand(ℓ.distribution)

"""
A function returning a log density and a gradient.
"""
struct FunctionLogDensity{F}
    dimension::Int
    f::F
end

dimension(ℓ::FunctionLogDensity) = length(ℓ.dimension)

logdensity_and_gradient(ℓ::FunctionLogDensity, x::AbstractVector) = ℓ.f(x)

"""
$(SIGNATURES)

A log density always returning a constant log density and gradient (thus not necessarily
consistent).
"""
function constant_logdensity(logdensity, gradient)
    FunctionLogDensity(length(gradient), _ -> (logdensity, gradient))
end

####
#### Hamiltonian test helper functions
####

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

find_stable_ϵ(H::Hamiltonian{<:Any,<: DistributionLogDensity}) =
    find_stable_ϵ(H.κ, cov(H.ℓ.distribution))

"Random Hamiltonian `H` with phasepoint `z`, with dimension `K`."
function rand_Hz(K)
    μ = randn(K)
    Σ = rand_Σ(K)
    κ = GaussianKineticEnergy(inv(rand_Σ(Diagonal, K)))
    dist = MvNormal(μ, Matrix(Σ))
    H = Hamiltonian(κ, DistributionLogDensity(dist))
    z = rand_phasepoint(RNG, H, rand(RNG, dist))
    H, z
end
