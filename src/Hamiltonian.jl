using Parameters

export
    KineticEnergy,
    EuclideanKE,
    GaussianKE,
    logdensity,
    loggradient

"""
Kinetic energy specifications.

For all subtypes, it is assumed that kinetic energy is symmetric in
the momentum `p`, ie.

```julia
logdensity(::KineticEnergy, p, q) == logdensity(::KineticEnergy, -p, q)
```

When the above is violated, various implicit assumptions will not hold.
"""
abstract type KineticEnergy end

"Euclidean kinetic energies (position independent)."
abstract type EuclideanKE <: KineticEnergy end

"""
Gaussian kinetic energy.

p|q ∼ Normal(0, M) (importantly, independent of q)

The square root of M⁻¹ is stored.
"""
struct GaussianKE{T <: Union{AbstractMatrix,UniformScaling}} <: EuclideanKE
    "U'U = M⁻¹"
    U::T
end

"Return U'*U*p."
invM_premultiply(κ::GaussianKE, p) = κ.U'*(κ.U*p)

"""
    logdensity(κ, p[, q])

Return the log density of kinetic energy `κ`, at momentum `p`. Some
kinetic energies (eg Riemannian geometry)  will need `q`, too.
"""
logdensity(κ::GaussianKE, p, q = nothing) = -sum(abs2, κ.U*p)/2

loggradient(κ::GaussianKE, p, q = nothing) = -invM_premultiply(κ, p)

propose(κ::GaussianKE, q = nothing) = κ.U \ randn(size(κ.U, 1 ))

"""
Specification for the Hamiltonian. Determines the kinetic and
potential energy, but not the not the actual algorithm.
"""
struct Hamiltonian{Tℓ, Tκ}
    "The (log) density we are sampling from."
    ℓ::Tℓ
    "The kinetic energy."
    κ::Tκ
end

"A point in phase space, consists of a position and a momentum."
struct PhasePoint{Tq,Tp}
    "Position."
    q::Tq
    "Momentum."
    p::Tp
end

"""
Log density for Hamiltonian `H` at point `z`.
"""
logdensity(H::Hamiltonian, z::PhasePoint) = logdensity(H.κ, z.p, z.q) + logdensity(H.ℓ, z.q)

"Take a leapfrog step in phase space."
function leapfrog{Tℓ, Tκ <: EuclideanKE}(H::Hamiltonian{Tℓ,Tκ}, z::PhasePoint, ϵ)
    @unpack ℓ, κ = H
    @unpack p, q = z
    p₊ = p + ϵ/2 * loggradient(ℓ, q)
    q′ = q - ϵ * loggradient(κ, p₊)
    p′ = p₊ + ϵ/2 * loggradient(ℓ, q′)
    PhasePoint(q′, p′)
end
