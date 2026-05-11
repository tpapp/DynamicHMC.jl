#####
##### stepsize heuristics and adaptation
#####

####
#### initial stepsize
####

"""
$(TYPEDEF)

Parameters for the search algorithm for the initial stepsize.

The algorithm finds an initial stepsize ``ϵ`` so that the local log acceptance ratio
``A(ϵ)`` is near `params.log_threshold`.

$FIELDS

!!! NOTE

    The algorithm is from Hoffman and Gelman (2014), default threshold modified to `0.8` following later practice in Stan.
"""
struct InitialStepsizeSearch{T<:Real}
    "The stepsize where the search is started."
    initial_ϵ::T
    "Log of the threshold that needs to be crossed."
    log_threshold::T
    "Maximum number of iterations for crossing the threshold."
    maxiter_crossing::Int
    function InitialStepsizeSearch(; log_threshold = log(0.8), initial_ϵ = 0.1, maxiter_crossing = 400)
        T = promote_type(typeof(log_threshold), typeof(initial_ϵ))
        @argcheck isfinite(log_threshold) && log_threshold < 0
        @argcheck isfinite(initial_ϵ) && 0 < initial_ϵ
        @argcheck maxiter_crossing ≥ 50
        new{T}(T(initial_ϵ), T(log_threshold), maxiter_crossing)
    end
end

"""
$(SIGNATURES)

Find an initial stepsize that matches the conditions of `parameters` (see
[`InitialStepsizeSearch`](@ref)).

`A` is the local log acceptance ratio (uncapped). Cf [`local_log_acceptance_ratio`](@ref).
"""
function find_initial_stepsize(parameters::InitialStepsizeSearch, A)
    (; initial_ϵ, log_threshold, maxiter_crossing) = parameters
    ϵ = initial_ϵ
    Aϵ = A(ϵ)
    double = Aϵ > log_threshold # do we double?
    for _ in 1:maxiter_crossing
        ϵ′ = double ? 2 * ϵ : ϵ / 2
        Aϵ′ = A(ϵ′)
        (double ? Aϵ′ < log_threshold : Aϵ′ > log_threshold) && return ϵ′
        ϵ = ϵ′
    end
    dir = double ? "below" : "above"
    _error("Initial stepsize search reached maximum number of iterations from $(dir) without crossing.";
           maxiter_crossing, initial_ϵ, ϵ)
end

"""
$(SIGNATURES)

Return a function of the stepsize (``ϵ``) that calculates the local log acceptance
ratio for a single leapfrog step around `z` along the Hamiltonian `H`. Formally,
let

```julia
A(ϵ) = logdensity(H, leapfrog(H, z, ϵ)) - logdensity(H, z)
```

Note that the ratio is not capped by `0`, so it is not a valid (log) probability *per se*.
"""
function local_log_acceptance_ratio(H, z)
    ℓ0 = logdensity(H, z)
    isfinite(ℓ0) ||
        _error("Starting point has non-finite density.";
               hamiltonian_logdensity = ℓ0, logdensity = z.Q.ℓq, position = z.Q.q)
    function(ϵ)
        z1 = leapfrog(H, z, ϵ)
        ℓ1 = logdensity(H, z1)
        ℓ1 - ℓ0
    end
end

"""
$(TYPEDEF)

Parameters for the dual averaging algorithm of Gelman and Hoffman (2014, Algorithm 6).

To get reasonable defaults, initialize with `DualAveraging()`.

# Fields

$(FIELDS)
"""
struct DualAveraging{T}
    "target acceptance rate"
    δ::T
    "regularization scale"
    γ::T
    "relaxation exponent"
    κ::T
    "offset"
    t₀::Int
    function DualAveraging(δ::T, γ::T, κ::T, t₀::Int) where {T <: Real}
        @argcheck 0 < δ < 1
        @argcheck γ > 0
        @argcheck 0.5 < κ ≤ 1
        @argcheck t₀ ≥ 0
        new{T}(δ, γ, κ, t₀)
    end
end

function DualAveraging(; δ = 0.8, γ = 0.05, κ = 0.75, t₀ = 10)
    DualAveraging(promote(δ, γ, κ)..., t₀)
end

"Current state of adaptation for `ϵ`."
Base.@kwdef struct DualAveragingState{T <: AbstractFloat}
    μ::T
    m::Int
    H̄::T
    logϵ::T
    logϵ̄::T
end

"""
$(SIGNATURES)

Return an initial adaptation state for the adaptation method and a stepsize `ϵ`.
"""
function initial_adaptation_state(::DualAveraging{T}, ϵ) where T <: AbstractFloat
    @argcheck ϵ > 0
    logϵ = log(ϵ)
    DualAveragingState{T}(; μ = log(T(10)) + logϵ, m = 1, H̄ = zero(logϵ), logϵ, logϵ̄ = zero(logϵ))
end

"""
$(SIGNATURES)

Update the adaptation `A` of log stepsize `logϵ` with average Metropolis acceptance rate `a`
over the whole visited trajectory, using the dual averaging algorithm of Gelman and Hoffman
(2014, Algorithm 6). Return the new adaptation state.
"""
function adapt_stepsize(parameters::DualAveraging, A::DualAveragingState, a)
    @argcheck 0 ≤ a ≤ 1
    (; δ, γ, κ, t₀) = parameters
    (; μ, m, H̄, logϵ, logϵ̄) = A
    m += 1
    H̄ += (δ - a - H̄) / (m + t₀)
    T_m = oftype(μ, m)
    logϵ = μ - sqrt(T_m)/γ * H̄
    logϵ̄ += T_m^(-κ)*(logϵ - logϵ̄)
    DualAveragingState(; μ, m, H̄, logϵ, logϵ̄)
end

"""
$(SIGNATURES)

Return the stepsize `ϵ` for the next HMC step while adapting.
"""
current_ϵ(A::DualAveragingState, tuning = true) = exp(A.logϵ)

"""
$(SIGNATURES)

Return the final stepsize `ϵ` after adaptation.
"""
final_ϵ(A::DualAveragingState, tuning = true) = exp(A.logϵ̄)

###
### fixed stepsize adaptation placeholder
###

"""
$(SIGNATURES)

Adaptation with fixed stepsize. Leaves `ϵ` unchanged.
"""
struct FixedStepsize end

initial_adaptation_state(::FixedStepsize, ϵ) = ϵ

adapt_stepsize(::FixedStepsize, ϵ, a) = ϵ

current_ϵ(ϵ::Real) = ϵ

final_ϵ(ϵ::Real) = ϵ

###
### type conversion helpers for warmup pipeline
###

_oftype(da::DualAveraging{T}, ::Type{T}) where {T} = da
_oftype(da::DualAveraging, ::Type{T}) where {T<:AbstractFloat} =
    DualAveraging(T(da.δ), T(da.γ), T(da.κ), da.t₀)

_oftype(iss::InitialStepsizeSearch{T}, ::Type{T}) where {T} = iss
_oftype(iss::InitialStepsizeSearch, ::Type{T}) where {T<:Real} =
    InitialStepsizeSearch(; log_threshold = T(iss.log_threshold),
                            initial_ϵ = T(iss.initial_ϵ),
                            maxiter_crossing = iss.maxiter_crossing)

_oftype(fs::FixedStepsize, ::Type) = fs
