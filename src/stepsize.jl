#####
##### stepsize heuristics and adaptation
#####

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
    function InitialStepsizeSearch(; a_min = 0.25, a_max = 0.75, ϵ₀ = 1.0,
                                   C = 2.0,
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

find_initial_stepsize(parameters::InitialStepsizeSearch, H, z) =
    find_initial_stepsize(parameters, local_acceptance_ratio(H, z))

"""
$(SIGNATURES)

Return a function of the stepsize (``ϵ``) that calculates the local acceptance
ratio for a single leapfrog step around `z` along the Hamiltonian `H`. Formally,
let

```math
A(ϵ) = \\exp(\\text{neg_energy}(H, \\text{leapfrog}(H, z, ϵ)) - \\text{neg_energy}(H, z))
```

Note that the ratio is not capped by `1`, so it is not a valid probability
*per se*.
"""
function local_acceptance_ratio(H, z)
    target = neg_energy(H, z)
    isfinite(target) ||
        throw(DomainError(z.p, "Starting point has non-finite density."))
    ϵ -> exp(neg_energy(H, leapfrog(H, z, ϵ)) - target)
end

"""
$(SIGNATURES)

Return a matrix of [`local_acceptance_ratio`](@ref) values for stepsizes `ϵs`
and the given momentums `ps`. The latter is calculated from random values when
an integer is given.

To facilitate plotting, ``-∞`` values are replaced by `NaN`.
"""
function explore_local_acceptance_ratios(H, q, ϵs, ps)
    R = hcat([local_acceptance_ratio(H, q, p).(ϵs) for p in ps]...)
    R[isinfinite.(R)] .= NaN
    R
end

explore_local_acceptance_ratios(H, q, ϵs, N::Int) =
    explore_local_acceptance_ratios(H, q, ϵs, [rand(H.κ) for _ in 1:N])

"""
Parameters for the dual averaging algorithm of Gelman and Hoffman (2014,
Algorithm 6).

To get reasonable defaults, initialize with
`DualAveragingParameters(logϵ₀)`. See [`adapting_ϵ`](@ref) for a joint
constructor.
"""
struct DualAveragingParameters{T}
    μ::T
    "target acceptance rate"
    δ::T
    "regularization scale"
    γ::T
    "relaxation exponent"
    κ::T
    "offset"
    t₀::Int
    function DualAveragingParameters{T}(μ, δ, γ, κ, t₀) where {T}
        @argcheck 0 < δ < 1
        @argcheck γ > 0
        @argcheck 0.5 < κ ≤ 1
        @argcheck t₀ ≥ 0
        new(μ, δ, γ, κ, t₀)
    end
end

DualAveragingParameters(μ::T, δ::T, γ::T, κ::T, t₀::Int) where T =
    DualAveragingParameters{T}(μ, δ, γ, κ, t₀)

DualAveragingParameters(logϵ₀; δ = 0.8, γ = 0.05, κ = 0.75, t₀ = 10) =
    DualAveragingParameters(promote(log(10) + logϵ₀, δ, γ, κ)..., t₀)

"Current state of adaptation for `ϵ`. Use `DualAverageingAdaptation(logϵ₀)` to
get an initial value. See [`adapting_ϵ`](@ref) for a joint constructor."
struct DualAveragingAdaptation{T <: AbstractFloat}
    m::Int
    H̄::T
    logϵ::T
    logϵ̄::T
end

"""
$(SIGNATURES)

Return the stepsize `ϵ` for the next HMC step while adapting.
"""
get_current_ϵ(A::DualAveragingAdaptation, tuning = true) = exp(A.logϵ)

"""
$(SIGNATURES)

Return the final stepsize `ϵ` after adaptation.
"""
get_final_ϵ(A::DualAveragingAdaptation, tuning = true) = exp(A.logϵ̄)

DualAveragingAdaptation(logϵ₀) =
    DualAveragingAdaptation(0, zero(logϵ₀), logϵ₀, zero(logϵ₀))

"""
    DA_params, A = $(SIGNATURES)

Constructor for both the adaptation parameters and the initial state.
"""
function adapting_ϵ(ϵ; args...)
    logϵ = log(ϵ)
    DualAveragingParameters(logϵ; args...), DualAveragingAdaptation(logϵ)
end

"""
    A′ = $(SIGNATURES)

Update the adaptation `A` of log stepsize `logϵ` with average Metropolis
acceptance rate `a` over the whole visited trajectory, using the dual averaging
algorithm of Gelman and Hoffman (2014, Algorithm 6). Return the new adaptation.
"""
function adapt_stepsize(parameters::DualAveragingParameters,
                        A::DualAveragingAdaptation, a)
    @argcheck 0 ≤ a ≤ 1
    @unpack μ, δ, γ, κ, t₀ = parameters
    @unpack m, H̄, logϵ, logϵ̄ = A
    m += 1
    H̄ += (δ - a - H̄) / (m+t₀)
    logϵ = μ - √m/γ*H̄
    logϵ̄ += m^(-κ)*(logϵ - logϵ̄)
    DualAveragingAdaptation(m, H̄, logϵ, logϵ̄)
end
