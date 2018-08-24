# This file contains building blocks for traversing a Hamiltonian
# deterministically, using the leapfrog integrator.

export KineticEnergy, EuclideanKE, GaussianKE

"""
Kinetic energy specifications.

For all subtypes, it is implicitly assumed that kinetic energy is symmetric in
the momentum `p`, ie.

```julia
neg_energy(::KineticEnergy, p, q) == neg_energy(::KineticEnergy, -p, q)
```

When the above is violated, the consequences are undefined.
"""
abstract type KineticEnergy end

"Euclidean kinetic energies (position independent)."
abstract type EuclideanKE <: KineticEnergy end

"""
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
    GaussianKE(M⁻¹::AbstractMatrix)

Gaussian kinetic energy with the given inverse covariance matrix `M⁻¹`.
"""
GaussianKE(Minv::AbstractMatrix) = GaussianKE(Minv, cholesky(inv(Minv)).L)

"""
    GaussianKE(N::Int, [m⁻¹ = 1.0])

Gaussian kinetic energy with a diagonal inverse covariance matrix `M⁻¹=m⁻¹*I`.
"""
GaussianKE(N::Int, m⁻¹ = 1.0) = GaussianKE(Diagonal(fill(m⁻¹, N)))

show(io::IO, κ::GaussianKE) =
    print(io::IO, "Gaussian kinetic energy, √diag(M⁻¹): $(.√(diag(κ.Minv)))")

"""
    neg_energy(κ, p, [q])

Return the log density of kinetic energy `κ`, at momentum `p`. Some kinetic
energies (eg Riemannian geometry) will need `q`, too.
"""
neg_energy(κ::GaussianKE, p, q = nothing) = -dot(p, κ.Minv * p) / 2

"""
    get_p♯(κ, p, [q])

Return ``p♯``, used for turn diagnostics.
"""
get_p♯(κ::GaussianKE, p, q = nothing) = κ.Minv * p

loggradient(κ::GaussianKE, p, q = nothing) = -get_p♯(κ, p)

rand(rng::AbstractRNG, κ::GaussianKE, q = nothing) = κ.W * randn(rng, size(κ.W, 1))

"""
    Hamiltonian(ℓ, κ)

Construct a Hamiltonian from the log density `ℓ`, and the kinetic energy
specification `κ`. Calls of `ℓ` with a vector are expected to return a value
that supports `DiffResults.value` and `DiffResults.gradient`.
"""
struct Hamiltonian{Tℓ, Tκ}
    "The (log) density we are sampling from."
    ℓ::Tℓ
    "The kinetic energy."
    κ::Tκ
end

show(io::IO, H::Hamiltonian) = print(io, "Hamiltonian with $(H.κ)")

"""
    is_valid_ℓq(ℓq)

Test that a value returned by ℓ is *valid*, in the following sense:

1. supports `DiffResults.value` and `DiffResults.gradient` (when not, a
`MethodError` is thrown),

2. the value is a float, either `-Inf` or finite,

3. the gradient is finite when the value is; otherwise the gradient is ignored.
"""
function is_valid_ℓq(ℓq)
    v = DiffResults.value(ℓq)
    v isa AbstractFloat || return false
    (v == -Inf) || (isfinite(v) && all(isfinite, DiffResults.gradient(ℓq)))
end

"""
A point in phase space, consists of a position and a momentum.

Log densities and gradients are saved for speed gains, so that the gradient of ℓ
at q is not calculated twice for every leapfrog step (both as start- and
endpoints).

Because of caching, a `PhasePoint` should only be used with a specific
Hamiltonian.
"""
struct PhasePoint{T,S}
    "Position."
    q::T
    "Momentum."
    p::T
    "ℓ(q). Cached for reuse in sampling."
    ℓq::S
    function PhasePoint(q::T, p::T, ℓq::S) where {T,S}
        @argcheck is_valid_ℓq(ℓq) DomainError("Invalid value of ℓ.")
        @argcheck length(p) == length(q)
        new{T,S}(q, p, ℓq)
    end
end

"""
    get_ℓq(z)

The value returned by `ℓ` when evaluated at position `q`.
"""
get_ℓq(z::PhasePoint) = z.ℓq

"""
    phasepoint_in(H::Hamiltonian, q, p)

The recommended interface for creating a phase point in a Hamiltonian. Computes
cached values.
"""
phasepoint_in(H::Hamiltonian, q, p) = PhasePoint(q, p, H.ℓ(q))

"""
    rand_phasepoint(rng, H, q)

Extend a position `q` to a phasepoint with a random momentum according to the
kinetic energy of `H`.
"""
rand_phasepoint(rng, H, q) = phasepoint_in(H, q, rand(rng, H.κ))

"""
    $SIGNATURES

Log density for Hamiltonian `H` at point `z`.

If `ℓ(q) == -Inf` (rejected), ignores the kinetic energy.
"""
function neg_energy(H::Hamiltonian, z::PhasePoint)
    v = DiffResults.value(get_ℓq(z))
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
    pₘ = p + ϵ/2 * DiffResults.gradient(ℓq)
    q′ = q - ϵ * loggradient(κ, pₘ)
    ℓq′ = ℓ(q′)
    p′ = pₘ + ϵ/2 * DiffResults.gradient(ℓq′)
    PhasePoint(q′, p′, ℓq′)
end
