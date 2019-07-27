####
#### general test environment
####

const RNG = Random.GLOBAL_RNG   # shorthand
Random.seed!(RNG, UInt32[0x23ef614d, 0x8332e05c, 0x3c574111, 0x121aa2f4])

"Tolerant testing in a CI environment."
const RELAX = (k = "CONTINUOUS_INTEGRATION"; haskey(ENV, k) && ENV[k] == "true")

"""
    $SIGNATURES

Random positive definite matrix of size `n` x `n` (for testing).
"""
function rand_Σ(::Type{Symmetric}, n)
    A = randn(n, n)
    Symmetric(A'*A .+ 0.01)
end

rand_Σ(::Type{Diagonal}, n) = Diagonal(randn(n).^2 .+ 0.01)

rand_Σ(n::Int) = rand_Σ(Symmetric, n)

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

dimension(ℓ::FunctionLogDensity) = length(ℓ.distribution)

function logdensity_and_gradient(ℓ::FunctionLogDensity, x::AbstractVector)
    ℓ.f(x)
end

"""
$(SIGNATURES)

A log density always returning a constant (for testing).
"""
function FunctionLogDensity(logdensity, gradient)
    FunctionLogDensity(length(gradient), _ -> (logdensity, gradient))
end

"Random Hamiltonian `H` with phasepoint `z`, with dimension `K`."
function rand_Hz(K)
    μ = randn(K)
    Σ = rand_Σ(K)
    κ = GaussianKE(inv(rand_Σ(Diagonal, K)))
    H = Hamiltonian(DistributionLogDensity(MvNormal(μ, Matrix(Σ))), κ)
    z = rand_phasepoint(RNG, H, μ)
    H, z
end
