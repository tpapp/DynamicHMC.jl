#####
##### stepsize heuristics and adaptation
#####

####
#### initial stepsize
####

"""
$(TYPEDEF)

Parameters for the search algorithm for the initial stepsize.

The algorithm finds an initial stepsize ``ϵ`` so that the local acceptance ratio
``A(ϵ)`` satisfies

```math
a_\\text{min} ≤ A(ϵ) ≤ a_\\text{max}
```

This is achieved by an initial bracketing, then bisection.

$FIELDS

!!! note

    Cf. Hoffman and Gelman (2014), which does not ensure bounds for the
    acceptance ratio, just that it has crossed a threshold. This version seems
    to work better for some tricky posteriors with high curvature.
"""
struct InitialStepsizeSearch
    "Lowest local acceptance rate."
    a_min::Float64
    "Highest local acceptance rate."
    a_max::Float64
    "Initial stepsize."
    ϵ₀::Float64
    "Scale factor for initial bracketing, > 1. *Default*: `2.0`."
    C::Float64
    "Maximum number of iterations for initial bracketing."
    maxiter_crossing::Int
    "Maximum number of iterations for bisection."
    maxiter_bisect::Int
    function InitialStepsizeSearch(; a_min = 0.25, a_max = 0.75, ϵ₀ = 1.0, C = 2.0,
                                   maxiter_crossing = 400, maxiter_bisect = 400)
        @argcheck 0 < a_min < a_max < 1
        @argcheck 0 < ϵ₀
        @argcheck 1 < C
        @argcheck maxiter_crossing ≥ 50
        @argcheck maxiter_bisect ≥ 50
        new(a_min, a_max, ϵ₀, C, maxiter_crossing, maxiter_bisect)
    end
end

"""
$(SIGNATURES)

Find the stepsize for which the local acceptance rate `A(ϵ)` crosses `a`.

Return `ϵ₀, A(ϵ₀), ϵ₁`, A(ϵ₁)`, where `ϵ₀` and `ϵ₁` are stepsizes before and
after crossing `a` with `A(ϵ)`, respectively.

Assumes that ``A(ϵ₀) ∉ (a_\\text{min}, a_\\text{max})``, where the latter are
defined in `parameters`.

- `parameters`: parameters for the iteration.

- `A`: local acceptance ratio (uncapped), a function of stepsize `ϵ`

- `ϵ₀`, `Aϵ₀`: initial value of `ϵ`, and `A(ϵ₀)`
"""
function find_crossing_stepsize(parameters, A, ϵ₀, Aϵ₀ = A(ϵ₀))
    @unpack a_min, a_max, C, maxiter_crossing = parameters
    s, a = Aϵ₀ > a_max ? (1.0, a_max) : (-1.0, a_min)
    if s < 0                    # when A(ϵ) < a,
        C = 1/C                 # decrease ϵ
    end
    for _ in 1:maxiter_crossing
        ϵ = ϵ₀ * C
        Aϵ = A(ϵ)
        if s*(Aϵ - a) ≤ 0
            return ϵ₀, Aϵ₀, ϵ, Aϵ
        else
            ϵ₀ = ϵ
            Aϵ₀ = Aϵ
        end
    end
    # should never each this, miscoded log density?
    dir = s > 0 ? "below" : "above"
    error("Reached maximum number of iterations searching for ϵ from $(dir).")
end

"""
$(SIGNATURES)

Return the desired stepsize `ϵ` by bisection.

- `parameters`: algorithm parameters, see [`InitialStepsizeSearch`](@ref)

- `A`: local acceptance ratio (uncapped), a function of stepsize `ϵ`

- `ϵ₀`, `ϵ₁`, `Aϵ₀`, `Aϵ₁`: stepsizes and acceptance rates (latter optional).

This function assumes that ``ϵ₀ < ϵ₁``, the stepsize is not yet acceptable, and
the cached `A` values have the correct ordering.
"""
function bisect_stepsize(parameters, A, ϵ₀, ϵ₁, Aϵ₀ = A(ϵ₀), Aϵ₁ = A(ϵ₁))
    @unpack a_min, a_max, maxiter_bisect = parameters
    @argcheck ϵ₀ < ϵ₁
    @argcheck Aϵ₀ > a_max && Aϵ₁ < a_min
    for _ in 1:maxiter_bisect
        ϵₘ = middle(ϵ₀, ϵ₁)
        Aϵₘ = A(ϵₘ)
        if a_min ≤ Aϵₘ ≤ a_max  # in
            return ϵₘ
        elseif Aϵₘ < a_min      # above
            ϵ₁ = ϵₘ
            Aϵ₁ = Aϵₘ
        else                    # below
            ϵ₀ = ϵₘ
            Aϵ₀ = Aϵₘ
        end
    end
    # should never each this, miscoded log density?
    error("Reached maximum number of iterations while bisecting interval for ϵ.")
end

"""
$(SIGNATURES)

Find an initial stepsize that matches the conditions of `parameters` (see
[`InitialStepsizeSearch`](@ref)).

`A` is the local acceptance ratio (uncapped). When given a Hamiltonian `H` and a
phasepoint `z`, it will be calculated using [`local_acceptance_ratio`](@ref).
"""
function find_initial_stepsize(parameters::InitialStepsizeSearch, A)
    @unpack a_min, a_max, ϵ₀ = parameters
    Aϵ₀ = A(ϵ₀)
    if a_min ≤ Aϵ₀ ≤ a_max
        ϵ₀
    else
        ϵ₀, Aϵ₀, ϵ₁, Aϵ₁ = find_crossing_stepsize(parameters, A, ϵ₀, Aϵ₀)
        if a_min ≤ Aϵ₁ ≤ a_max  # in interval
            ϵ₁
        elseif ϵ₀ < ϵ₁          # order as necessary
            bisect_stepsize(parameters, A, ϵ₀, ϵ₁, Aϵ₀, Aϵ₁)
        else
            bisect_stepsize(parameters, A, ϵ₁, ϵ₀, Aϵ₁, Aϵ₀)
        end
    end
end

"""
$(SIGNATURES)

Uncapped log acceptance ratio of a Langevin step.
"""
function log_acceptance_ratio(H, z, ϵ)
    target = logdensity(H, z)
    isfinite(target) || throw(DomainError(z, "Starting point has non-finite density."))
    logdensity(H, leapfrog(H, z, ϵ)) - target
end

"""
$(SIGNATURES)

Return a function of the stepsize (``ϵ``) that calculates the local acceptance
ratio for a single leapfrog step around `z` along the Hamiltonian `H`. Formally,
let

```julia
A(ϵ) = exp(logdensity(H, leapfrog(H, z, ϵ)) - logdensity(H, z))
```

Note that the ratio is not capped by `1`, so it is not a valid probability *per se*.
"""
function local_acceptance_ratio(H, z)
    target = logdensity(H, z)
    isfinite(target) ||
        throw(DomainError(z.p, "Starting point has non-finite density."))
    ϵ -> exp(logdensity(H, leapfrog(H, z, ϵ)) - target)
end

function find_initial_stepsize(parameters::InitialStepsizeSearch, H, z)
    find_initial_stepsize(parameters, local_acceptance_ratio(H, z))
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
struct DualAveragingState{T <: AbstractFloat}
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
function initial_adaptation_state(::DualAveraging, ϵ)
    @argcheck ϵ > 0
    logϵ = log(ϵ)
    DualAveragingState(log(10) + logϵ, 0, zero(logϵ), logϵ, zero(logϵ))
end

"""
$(SIGNATURES)

Update the adaptation `A` of log stepsize `logϵ` with average Metropolis acceptance rate `a`
over the whole visited trajectory, using the dual averaging algorithm of Gelman and Hoffman
(2014, Algorithm 6). Return the new adaptation state.
"""
function adapt_stepsize(parameters::DualAveraging, A::DualAveragingState, a)
    @argcheck 0 ≤ a ≤ 1
    @unpack δ, γ, κ, t₀ = parameters
    @unpack μ, m, H̄, logϵ, logϵ̄ = A
    m += 1
    H̄ += (δ - a - H̄) / (m + t₀)
    logϵ = μ - √m/γ * H̄
    logϵ̄ += m^(-κ)*(logϵ - logϵ̄)
    DualAveragingState(μ, m, H̄, logϵ, logϵ̄)
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
