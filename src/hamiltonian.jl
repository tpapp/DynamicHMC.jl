#####
##### Building blocks for traversing a Hamiltonian deterministically, using the leapfrog
##### integrator.
#####

export KineticEnergy, EuclideanKE, GaussianKE

"""
$(TYPEDEF)

Kinetic energy specifications.

For all subtypes, it is implicitly assumed that kinetic energy is symmetric in
the momentum `p`, ie.

```julia
neg_energy(::KineticEnergy, p, q) == neg_energy(::KineticEnergy, -p, q)
```

When the above is violated, the consequences are undefined.
"""
abstract type KineticEnergy end

"""
$(TYPEDEF)

Euclidean kinetic energies (position independent).
"""
abstract type EuclideanKE <: KineticEnergy end

"""
$(TYPEDEF)

Gaussian kinetic energy.

```math
p ∣ q ∼ N(0, M)
```

**independently** of ``q``.

The inverse covariance ``M⁻¹`` is stored.
"""
struct GaussianKE{T <: AbstractMatrix, S <: AbstractMatrix} <: EuclideanKE
    "M⁻¹"
    Minv::T
    "W such that W*W'=M. Used for generating random draws."
    W::S
    function GaussianKE{T, S}(Minv, W) where {T, S}
        @argcheck checksquare(Minv) == checksquare(W)
        new(Minv, W)
    end
end

GaussianKE(M::T, W::S) where {T,S} = GaussianKE{T,S}(M, W)

"""
$(SIGNATURES)

Gaussian kinetic energy with the given inverse covariance matrix `M⁻¹`.
"""
GaussianKE(Minv::AbstractMatrix) = GaussianKE(Minv, cholesky(inv(Minv)).L)

"""
$(SIGNATURES)

Gaussian kinetic energy with a diagonal inverse covariance matrix `M⁻¹=m⁻¹*I`.
"""
GaussianKE(N::Int, m⁻¹ = 1.0) = GaussianKE(Diagonal(fill(m⁻¹, N)))

show(io::IO, κ::GaussianKE) =
    print(io::IO, "Gaussian kinetic energy, √diag(M⁻¹): $(.√(diag(κ.Minv)))")

"""
$(SIGNATURES)

Return the log density of kinetic energy `κ`, at momentum `p`. Some kinetic
energies (eg Riemannian geometry) will need `q`, too.
"""
neg_energy(κ::GaussianKE, p, q = nothing) = -dot(p, κ.Minv * p) / 2

"""
$(SIGNATURES)

Return ``p♯``, used for turn diagnostics.
"""
get_p♯(κ::GaussianKE, p, q = nothing) = κ.Minv * p

"""
$(SIGNATURES)

Calculate the gradient of the logarithm of kinetic energy at momentum `p` and
position `q`; the latter is ignored for Gaussian kinetic energies.
"""
loggradient(κ::GaussianKE, p, q = nothing) = -get_p♯(κ, p)

rand(rng::AbstractRNG, κ::GaussianKE, q = nothing) = κ.W * randn(rng, size(κ.W, 1))

"""
    Hamiltonian(ℓ, κ)

Construct a Hamiltonian from the log density `ℓ`, and the kinetic energy
specification `κ`. Calls of `ℓ` with a vector are expected to return a value
that supports `DiffResults.value` and `DiffResults.gradient`.
"""
struct Hamiltonian{Tℓ, Tκ}
    """
    The (log) density we are sampling from. Supports the `AbstractLogDensityProblem`
    interface, but it does not have to be a subtype.
    """
    ℓ::Tℓ
    "The kinetic energy."
    κ::Tκ
end

show(io::IO, H::Hamiltonian) = print(io, "Hamiltonian with $(H.κ)")

"""
$(TYPEDEF)

A point in phase space, consists of a position and a momentum.

Log densities and gradients are saved for speed gains, so that the gradient of ℓ
at q is not calculated twice for every leapfrog step (both as start- and
endpoints).

Because of caching, a `PhasePoint` should only be used with a specific
Hamiltonian.
"""
struct PhasePoint{T,S <: ValueGradient}
    "Position."
    q::T
    "Momentum."
    p::T
    "ℓ(q). Cached for reuse in sampling."
    ℓq::S
    function PhasePoint(q::T, p::T, ℓq::S) where {T,S}
        @argcheck length(p) == length(q) == length(ℓq.gradient)
        new{T,S}(q, p, ℓq)
    end
end

"""
    phasepoint_in(H::Hamiltonian, q, p)

The recommended interface for creating a phase point in a Hamiltonian. Computes
cached values.
"""
phasepoint_in(H::Hamiltonian, q, p) = PhasePoint(q, p, logdensity(ValueGradient, H.ℓ, q))

"""
$(SIGNATURES)

Extend a position `q` to a phasepoint with a random momentum according to the
kinetic energy of `H`.
"""
rand_phasepoint(rng::AbstractRNG, H, q) = phasepoint_in(H, q, rand(rng, H.κ))

"""
    $SIGNATURES

Log density for Hamiltonian `H` at point `z`.

If `ℓ(q) == -Inf` (rejected), ignores the kinetic energy.
"""
function neg_energy(H::Hamiltonian, z::PhasePoint)
    v = z.ℓq.value
    v == -Inf ? v : (v + neg_energy(H.κ, z.p, z.q))
end

get_p♯(H::Hamiltonian, z::PhasePoint) = get_p♯(H.κ, z.p, z.q)

"""
    leapfrog(H, z, ϵ)

Take a leapfrog step of length `ϵ` from `z` along the Hamiltonian `H`.

Return the new position.

The leapfrog algorithm uses the gradient of the next position to evolve the
momentum. If this is not finite, the momentum won't be either. Since the
constructor `PhasePoint` validates its arguments, this can only happen for
divergent points anyway, and should not cause a problem.
"""
function leapfrog(H::Hamiltonian{Tℓ,Tκ}, z::PhasePoint, ϵ) where {Tℓ, Tκ <: EuclideanKE}
    @unpack ℓ, κ = H
    @unpack p, q, ℓq = z
    pₘ = p + ϵ/2 * ℓq.gradient
    q′ = q - ϵ * loggradient(κ, pₘ)
    ℓq′ = logdensity(ValueGradient, ℓ, q′)
    p′ = pₘ + ϵ/2 * ℓq′.gradient
    PhasePoint(q′, p′, ℓq′)
end
