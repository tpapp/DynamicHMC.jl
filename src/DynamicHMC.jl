"""
Notation follows Betancourt (2017), with some differences

## Density
- `q` is the parameter vector
- `π(q)` is the density we are sampling from
- V(q) = -log(π(q)) ≡ -ℓ(q), in practice we use ℓ

## Momentym
- p is the momentum
- K(p,q) = -log π(p|q) is minus log conditional density for the momentum
- in practice we use ω for the momentum
"""
module DynamicHMC

using Parameters

import Base: rand

export
    KineticEnergy,
    EuclideanKE,
    GaussianKE,
    logdensity,
    loggradient,
    propose,
    HMC,
    leapfrog

"Kinetic energy specifications."
abstract type KineticEnergy end

"Euclidean kinetic energies (position independent)."
abstract type EuclideanKE <: KineticEnergy end

"""
Gaussian kinetic energy.

p|q ∼ Normal(0, M) (importantly, independent of q)

The square root M⁻¹ is stored.
"""
struct GaussianKE{T <: Union{AbstractMatrix,UniformScaling}} <: EuclideanKE
    "U'U = M⁻¹"
    U::T
end

logdensity(κ::GaussianKE, p, q = nothing) = -sum(abs2, κ.U*p)/2

loggradient(κ::GaussianKE, p, q = nothing) = -κ.U'*(κ.U*p)

propose(κ::GaussianKE, q = nothing) = κ.U \ randn(size(κ.U, 1 ))

"""
Specification for Hamiltonian Monte Carlo. Determines the kinetic and
potential energy, and the stepsize, but not the actual algorithm.
"""
struct HMC{Tℓ, Tκ, Tϵ}
    "The (log) density we are sampling from."
    ℓ::Tℓ
    "The kinetic energy."
    κ::Tκ
    "Stepsize for integration."
    ϵ::Tϵ
end

logdensity(hmc::HMC, q, p) = logdensity(hmc.κ, p, q) + logdensity(hmc.ℓ, q)

function leapfrog{Tℓ, Tκ <: EuclideanKE, Tϵ}(hmc::HMC{Tℓ,Tκ,Tϵ}, q, p)
    @unpack ℓ, κ, ϵ = hmc
    p₊ = p + ϵ/2 * loggradient(ℓ, q)
    q′ = q - ϵ * loggradient(κ, p₊)
    p′ = p₊ + ϵ/2 * loggradient(ℓ, q′)
    q′, p′
end

end # module
