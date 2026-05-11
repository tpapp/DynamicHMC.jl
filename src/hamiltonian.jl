#####
##### Building blocks for traversing a Hamiltonian deterministically, using the leapfrog
##### integrator.
#####

export GaussianKineticEnergy

####
#### kinetic energy
####

"""
$(TYPEDEF)

Kinetic energy specifications. Implements the methods

- `Base.size`

- [`kinetic_energy`](@ref)

- [`calculate_p♯`](@ref)

- [`∇kinetic_energy`](@ref)

- [`rand_p`](@ref)

For all subtypes, it is implicitly assumed that kinetic energy is symmetric in
the momentum `p`,

```julia
kinetic_energy(κ, p, q) == kinetic_energy(κ, .-p, q)
```

When the above is violated, the consequences are undefined.
"""
abstract type KineticEnergy end

"""
$(TYPEDEF)

Euclidean kinetic energies (position independent).
"""
abstract type EuclideanKineticEnergy <: KineticEnergy end

"""
$(TYPEDEF)

Gaussian kinetic energy, with ``K(q,p) = p ∣ q ∼ 1/2 pᵀ⋅M⁻¹⋅p + log|M|`` (without constant),
which is independent of ``q``.

The inverse covariance ``M⁻¹`` is stored.

!!! note
    Making ``M⁻¹`` approximate the posterior variance is a reasonable starting point.
"""
struct GaussianKineticEnergy{T <: AbstractMatrix,
                             S <: AbstractMatrix} <: EuclideanKineticEnergy
    "M⁻¹"
    M⁻¹::T
    "W such that W*W'=M. Used for generating random draws."
    W::S
    function GaussianKineticEnergy(M⁻¹::T, W::S) where {T, S}
        @argcheck checksquare(M⁻¹) == checksquare(W)
        new{T,S}(M⁻¹, W)
    end
end

"""
$(SIGNATURES)

Gaussian kinetic energy with the given inverse covariance matrix `M⁻¹`.
"""
GaussianKineticEnergy(M⁻¹::AbstractMatrix) = GaussianKineticEnergy(M⁻¹, cholesky(inv(M⁻¹)).L)

"""
$(SIGNATURES)

Gaussian kinetic energy with the given inverse covariance matrix `M⁻¹`.
"""
GaussianKineticEnergy(M⁻¹::Diagonal) = GaussianKineticEnergy(M⁻¹, Diagonal(.√inv.(diag(M⁻¹))))

"""
$(SIGNATURES)

Gaussian kinetic energy with a diagonal inverse covariance matrix `M⁻¹=m⁻¹*I`.
"""
GaussianKineticEnergy(N::Integer, m⁻¹ = 1.0) = GaussianKineticEnergy(Diagonal(Fill(float(m⁻¹), N)))

function Base.show(io::IO, κ::GaussianKineticEnergy{T}) where {T}
    print(io::IO, "Gaussian kinetic energy ($(nameof(T))), √diag(M⁻¹): $(.√(diag(κ.M⁻¹)))")
end

## NOTE about implementation: the 3 methods are callable without a third argument (`q`)
## because they are defined for Gaussian (Euclidean) kinetic energies.

Base.size(κ::GaussianKineticEnergy, args...) = size(κ.M⁻¹, args...)

"""
$(SIGNATURES)

Return kinetic energy `κ`, at momentum `p`.
"""
kinetic_energy(κ::GaussianKineticEnergy, p, q = nothing) = dot(p, κ.M⁻¹ * p) / 2

"""
$(SIGNATURES)

Return ``p♯ = M⁻¹⋅p``, used for turn diagnostics.
"""
calculate_p♯(κ::GaussianKineticEnergy, p, q = nothing) = κ.M⁻¹ * p

"""
$(SIGNATURES)

Calculate the gradient of the logarithm of kinetic energy in momentum `p`.
"""
∇kinetic_energy(κ::GaussianKineticEnergy, p, q = nothing) = calculate_p♯(κ, p)

"""
$(SIGNATURES)

Generate a random momentum from a kinetic energy at position `q`.
"""
rand_p(rng::AbstractRNG, κ::GaussianKineticEnergy, q = nothing) = κ.W * randn(rng, eltype(κ.W), size(κ.W, 1))

####
#### Hamiltonian
####

struct Hamiltonian{K,L}
    "The kinetic energy specification."
    κ::K
    """
    The (log) density we are sampling from. Supports the `LogDensityProblem` API.
    Technically, it is the negative of the potential energy.
    """
    ℓ::L
    """
    $(SIGNATURES)

    Construct a Hamiltonian from the log density `ℓ`, and the kinetic energy specification
    `κ`. `ℓ` with a vector are expected to support the `LogDensityProblems` API, with
    gradients.
    """
    function Hamiltonian(κ::K, ℓ::L) where {K <: KineticEnergy,L}
        @argcheck capabilities(ℓ) ≥ LogDensityOrder(1)
        @argcheck dimension(ℓ) == size(κ, 1)
        new{K,L}(κ, ℓ)
    end
end

Base.show(io::IO, H::Hamiltonian) = print(io, "Hamiltonian with $(H.κ)")

"""
$(TYPEDEF)

A log density evaluated at position `q`. The log densities and gradient are saved, so that
they are not calculated twice for every leapfrog step (both as start- and endpoints).

Because of caching, a `EvaluatedLogDensity` should only be used with a specific Hamiltonian,
preferably constructed with the `evaluate_ℓ` constructor.

In composite types and arguments, `Q` is usually used for this type.
"""
struct EvaluatedLogDensity{T,S}
    "Position."
    q::T
    "ℓ(q). Saved for reuse in sampling."
    ℓq::S
    "∇ℓ(q). Cached for reuse in sampling."
    ∇ℓq::T
    function EvaluatedLogDensity(q::T, ℓq::S, ∇ℓq::T) where {T <: AbstractVector,S <: Real}
        @argcheck length(q) == length(∇ℓq)
        new{T,S}(q, ℓq, ∇ℓq)
    end
end

# general constructors below are necessary to sanitize input from eg Diagnostics, or an
# initial position given as integers, etc

function EvaluatedLogDensity(q::AbstractVector, ℓq::Real, ∇ℓq::AbstractVector)
    q, ∇ℓq = promote(q, ∇ℓq)
    EvaluatedLogDensity(q, ℓq, ∇ℓq)
end

EvaluatedLogDensity(q, ℓq::Real, ∇ℓq) = EvaluatedLogDensity(collect(q), ℓq, collect(∇ℓq))

"""
$(SIGNATURES)

Evaluate log density and gradient and save with the position. Preferred interface for
creating `EvaluatedLogDensity` instances.

Non-finite elements in `q` always throw an error.

Non-finite and not `-Inf` elements in the log density throw an error if `strict`, otherwise
replace the log density with `-Inf`.

Non-finite elements in the gradient throw an error if `strict`, otherwise replace
the log density with `-Inf`.
"""
function evaluate_ℓ(ℓ, q; strict::Bool = false)
    all(isfinite, q) || _error("Position vector has non-finite elements."; q)
    ℓq, ∇ℓq = logdensity_and_gradient(ℓ, q)
    if (isfinite(ℓq) && all(isfinite, ∇ℓq)) || ℓq == -Inf
        # everything is finite, or log density is -Inf, which will be rejected
        EvaluatedLogDensity(q, ℓq, ∇ℓq)
    elseif !strict
        # something went wrong, but proceed and replace log density with -Inf, so it is
        # rejected.
        EvaluatedLogDensity(q, oftype(ℓq, -Inf), ∇ℓq) # somew
    elseif isfinite(ℓq)
        _error("Gradient has non-finite elements."; q, ∇ℓq)
    else
        _error("Invalid log posterior."; q, ℓq)
    end
end

"""
$(TYPEDEF)

A point in phase space, consists of a position (in the form of an evaluated log density `ℓ`
at `q`) and a momentum.
"""
struct PhasePoint{T <: EvaluatedLogDensity,S}
    "Evaluated log density."
    Q::T
    "Momentum."
    p::S
    function PhasePoint(Q::T, p::S) where {T,S}
        @argcheck length(p) == length(Q.q)
        new{T,S}(Q, p)
    end
end

"""
$(SIGNATURES)

Log density for Hamiltonian `H` at point `z`.

If `ℓ(q) == -Inf` (rejected), skips the kinetic energy calculation.

Non-finite values (incl `NaN`, `Inf`) are automatically converted to `-Inf`. This can happen
if

1. the log density is not a finite value,

2. the kinetic energy is not a finite value (which usually happens when `NaN` or `Inf` got
mixed in the leapfrog step, leading to an invalid position).
"""
function logdensity(H::Hamiltonian{<:EuclideanKineticEnergy}, z::PhasePoint)
    (; ℓq) = z.Q
    isfinite(ℓq) || return oftype(ℓq, -Inf)
    K = kinetic_energy(H.κ, z.p)
    ℓq - (isfinite(K) ? K : oftype(K, Inf))
end

function calculate_p♯(H::Hamiltonian{<:EuclideanKineticEnergy}, z::PhasePoint)
    calculate_p♯(H.κ, z.p)
end

"""
    leapfrog(H, z, ϵ)

Take a leapfrog step of length `ϵ` from `z` along the Hamiltonian `H`.

Return the new phase point.

The leapfrog algorithm uses the gradient of the next position to evolve the momentum. If
this is not finite, the momentum won't be either, `logdensity` above will catch this and
return an `-Inf`, making the point divergent.
"""
function leapfrog(H::Hamiltonian{<: EuclideanKineticEnergy}, z::PhasePoint, ϵ)
    (; ℓ, κ) = H
    (; p, Q) = z
    @argcheck isfinite(Q.ℓq) "Internal error: leapfrog called from non-finite log density"
    pₘ = p + ϵ/2 * Q.∇ℓq
    q′ = Q.q + ϵ * ∇kinetic_energy(κ, pₘ)
    Q′ = evaluate_ℓ(H.ℓ, q′)
    p′ = pₘ + ϵ/2 * Q′.∇ℓq
    PhasePoint(Q′, p′)
end
