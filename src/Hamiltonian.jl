using PDMats

import Base: rand

export
    logdensity, loggradient,
    Hamiltonian,
    KineticEnergy, EuclideanKE, GaussianKE,
    PhasePoint, phasepoint

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

p | q ∼ Normal(0, M)     (importantly, independent of q)

The inverse M⁻¹ is stored.
"""
struct GaussianKE{T <: AbstractPDMat} <: EuclideanKE
    "M⁻¹"
    Minv::T
end

"""
    logdensity(κ, p[, q])

Return the log density of kinetic energy `κ`, at momentum `p`. Some
kinetic energies (eg Riemannian geometry)  will need `q`, too.
"""
logdensity(κ::GaussianKE, p, q = nothing) = -quad(κ.Minv, p) / 2

getp♯(κ::GaussianKE, p, q = nothing) = κ.Minv * p

loggradient(κ::GaussianKE, p, q = nothing) = -getp♯(κ, p)

rand(rng, κ::GaussianKE, q = nothing) = whiten(κ.Minv, randn(rng, dim(κ.Minv)))

"""
    Hamiltonian(ℓ, κ)

Construct a Hamiltonian from the posterior density `ℓ`, and the
kinetic energy specification `κ`.
"""
struct Hamiltonian{Tℓ, Tκ}
    "The (log) density we are sampling from."
    ℓ::Tℓ
    "The kinetic energy."
    κ::Tκ
end

"""
A point in phase space, consists of a position and a momentum.

Log densities and gradients may be saved for speed gains, so a
`PhasePoint` should only be used with a specific Hamiltonian.
"""
struct PhasePoint{Tv,Tf}
    "Position."
    q::Tv
    "Momentum."
    p::Tv
    "Gradient of ℓ at q. Cached for reuse in leapfrog."
    ∇ℓq::Tv
    "ℓ at q. Cached for reuse in sampling."
    ℓq::Tf
end

"""
    phasepoint(H, q, p)

Preferred constructor for phasepoints, computes cached information.
"""
phasepoint(H, q, p) = PhasePoint(q, p, loggradient(H.ℓ, q), logdensity(H.ℓ, q))

"""
    rand_phasepoint(rng, H, q)

Extend a position `q` to a phasepoint with a random momentum according
to the kinetic energy of `H`.
"""
rand_phasepoint(rng, H, q) = phasepoint(H, q, rand(rng, H.κ))
    
"""
Log density for Hamiltonian `H` at point `z`.
"""
logdensity(H::Hamiltonian, z::PhasePoint) = z.ℓq + logdensity(H.κ, z.p, z.q)

getp♯(H::Hamiltonian, z::PhasePoint) = getp♯(H.κ, z.p, z.q)

"Take a leapfrog step in phase space."
function leapfrog{Tℓ, Tκ <: EuclideanKE}(H::Hamiltonian{Tℓ,Tκ}, z::PhasePoint, ϵ)
    @unpack ℓ, κ = H
    @unpack p, q, ∇ℓq = z
    pₘ = p + ϵ/2 * ∇ℓq
    q′ = q - ϵ * loggradient(κ, pₘ)
    ∇ℓq′ = loggradient(ℓ, q′)
    p′ = pₘ + ϵ/2 * ∇ℓq′
    PhasePoint(q′, p′, ∇ℓq′, logdensity(ℓ, q))
end
