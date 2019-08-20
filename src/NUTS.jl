#####
##### NUTS tree sampler implementation.
#####

####
#### Trajectory and implementation
####

"""
Representation of a trajectory, ie a Hamiltonian with a discrete integrator that
also checks for divergence.
"""
struct TrajectoryNUTS{TH,Tf,S}
    "Hamiltonian."
    H::TH
    "Log density of z (negative log energy) at initial point."
    π₀::Tf
    "Stepsize for leapfrog."
    ϵ::Tf
    "Smallest decrease allowed in the log density."
    min_Δ::Tf
    "Turn statistic configuration."
    turn_statistic_configuration::S
end

function move(trajectory::TrajectoryNUTS, z, fwd)
    @unpack H, ϵ = trajectory
    leapfrog(H, z, fwd ? ϵ : -ϵ)
end

###
### proposals
###

"""
$(SIGNATURES)

Random boolean which is `true` with the given probability `prob`.
"""
rand_bool(rng::AbstractRNG, prob::T) where {T <: AbstractFloat} = rand(rng, T) ≤ prob

function calculate_logprob2(::TrajectoryNUTS, is_doubling, ω₁, ω₂, ω)
    biased_progressive_logprob2(is_doubling, ω₁, ω₂, ω)
end

function combine_proposals(rng, ::TrajectoryNUTS, z₁, z₂, logprob2::Real, is_forward)
    (logprob2 ≥ 0 || rand_bool(rng, exp(logprob2))) ? z₂ : z₁
end

###
### statistics for visited nodes
###

struct AcceptanceStatistic{T}
    """
    Logarithm of the sum of metropolis acceptances probabilities over the whole trajectory
    (including invalid parts).
    """
    log_sum_α::T
    "Total number of leapfrog steps."
    steps::Int
end

function combine_acceptance_statistics(A::AcceptanceStatistic, B::AcceptanceStatistic)
    AcceptanceStatistic(logaddexp(A.log_sum_α, B.log_sum_α), A.steps + B.steps)
end

"""
$(SIGNATURES)

Acceptance statistic for a leaf. The initial leaf is considered not to be visited.
"""
function leaf_acceptance_statistic(Δ, is_initial)
    is_initial ? AcceptanceStatistic(oftype(Δ, -Inf), 0) : AcceptanceStatistic(min(Δ, 0), 1)
end

"""
$(SIGNATURES)

Return the acceptance rate (a `Real` betwen `0` and `1`).
"""
acceptance_rate(A::AcceptanceStatistic) = min(exp(A.log_sum_α) / A.steps, 1)

combine_visited_statistics(::TrajectoryNUTS, v, w) = combine_acceptance_statistics(v, w)

###
### turn analysis
###

"Statistics for the identification of turning points. See Betancourt (2017, appendix)."
struct GeneralizedTurnStatistic{T}
    p♯₋::T
    p♯₊::T
    ρ::T
end

function leaf_turn_statistic(::Val{:generalized}, H, z)
    p♯ = calculate_p♯(H, z)
    GeneralizedTurnStatistic(p♯, p♯, z.p)
end

function combine_turn_statistics(::TrajectoryNUTS,
                                 x::GeneralizedTurnStatistic, y::GeneralizedTurnStatistic)
    GeneralizedTurnStatistic(x.p♯₋, y.p♯₊, x.ρ + y.ρ)
end

function is_turning(::TrajectoryNUTS, τ::GeneralizedTurnStatistic)
    # Uses the generalized NUTS criterion from Betancourt (2017).
    @unpack p♯₋, p♯₊, ρ = τ
    @argcheck p♯₋ ≢ p♯₊ "internal error: is_turning called on a leaf"
    dot(p♯₋, ρ) < 0 || dot(p♯₊, ρ) < 0
end

###
### leafs
###

function leaf(trajectory::TrajectoryNUTS, z, is_initial)
    @unpack H, π₀, min_Δ, turn_statistic_configuration = trajectory
    Δ = is_initial ? zero(π₀) : logdensity(H, z) - π₀
    isdiv = Δ < min_Δ
    v = leaf_acceptance_statistic(Δ, is_initial)
    if isdiv
        nothing, v
    else
        τ = leaf_turn_statistic(turn_statistic_configuration, H, z)
        (z, Δ, τ), v
    end
end

####
#### NUTS interface
####

"Default maximum depth for trees."
const DEFAULT_MAX_TREE_DEPTH = 10

"""
$(TYPEDEF)

Options for building NUTS trees. These are the parameters that are expected to remain stable
during adaptation and sampling.

# Fields

$(FIELDS)
"""
struct TreeOptionsNUTS{S}
    "Maximum tree depth."
    max_depth::Int
    "Threshold for negative energy relative to starting point that indicated divergence."
    min_Δ::Float64
    """
    Turn statistic configuration. Currently only `Val(:generalized)` (the default) is
    supported.
    """
    turn_statistic_configuration::S
    function TreeOptionsNUTS(; max_depth = DEFAULT_MAX_TREE_DEPTH,
                             min_Δ = -1000.0,
                             turn_statistic_configuration = Val{:generalized}())
        @argcheck 0 < max_depth ≤ MAX_DIRECTIONS_DEPTH
        @argcheck min_Δ < 0
        S = typeof(turn_statistic_configuration)
        new{S}(Int(max_depth), Float64(min_Δ), turn_statistic_configuration)
    end
end

"""
$(TYPEDEF)

Diagnostic information for a single tree built with the No-U-turn sampler.

# Fields

Accessing fields directly is part of the API.

$(FIELDS)
"""
struct TreeStatisticsNUTS
    "Log density (negative energy)."
    π::Float64
    "Depth of the tree."
    depth::Int
    "Reason for termination. See [`InvalidTree`](@ref) and [`REACHED_MAX_DEPTH`](@ref)."
    termination::InvalidTree
    "Acceptance rate statistic."
    acceptance_rate::Float64
    "Number of leapfrog steps evaluated."
    steps::Int
    "Directions for tree doubling (useful for debugging)."
    directions::Directions
end

"""
$(SIGNATURES)

No-U-turn Hamiltonian Monte Carlo transition, using Hamiltonian `H`, starting at evaluated
log density position `Q`, using stepsize `ϵ`. `options` govern the details of tree
construction.

Return two values, the new evaluated log density position, and tree statistics.
"""
function NUTS_sample_tree(rng, options::TreeOptionsNUTS, H::Hamiltonian,
                          Q::EvaluatedLogDensity, ϵ;
                          p = rand_p(rng, H.κ), directions = rand(rng, Directions))
    @unpack max_depth, min_Δ, turn_statistic_configuration = options
    z = PhasePoint(Q, p)
    trajectory = TrajectoryNUTS(H, logdensity(H, z), ϵ, min_Δ, turn_statistic_configuration)
    ζ, v, termination, depth = sample_trajectory(rng, trajectory, z, max_depth, directions)
    tree_statistics = TreeStatisticsNUTS(logdensity(H, ζ), depth, termination,
                                         acceptance_rate(v), v.steps, directions)
    ζ.Q, tree_statistics
end
