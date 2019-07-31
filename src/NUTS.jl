#####
##### NUTS tree sampler implementation.
#####
##### Only TransitionNUTS and the exported functions are part of the API.
#####

export TreeStatisticsNUTS

####
#### Trajectory and implementation
####

"""
Representation of a trajectory, ie a Hamiltonian with a discrete integrator that
also checks for divergence.
"""
struct TrajectoryNUTS{TH,Tf}
    "Hamiltonian."
    H::TH
    "Log density of z (negative log energy) at initial point."
    π₀::Tf
    "Stepsize for leapfrog."
    ϵ::Tf
    "Smallest decrease allowed in the log density."
    min_Δ::Tf
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
### divergence statistics
###

"""
Divergence and acceptance statistics.

Calculated over all visited phase points (not just the tree that is sampled from).
"""
struct DivergenceStatistic{Tf}
    "`true` iff the sampler was terminated because of divergence."
    divergent::Bool
    """
    Sum of metropolis acceptances probabilities over the whole trajectory (including invalid
    parts).
    """
    ∑a::Tf
    "Total number of leapfrog steps."
    steps::Int
end

"""
$(SIGNATURES)

Empty divergence statistic (for initial node).
"""
divergence_statistic() = DivergenceStatistic(false, 0.0, 0)

"""
$(SIGNATURES)

Divergence statistic for leaves. `Δ` is the log density relative to the initial
point.
"""
function divergence_statistic(is_divergent, Δ)
    DivergenceStatistic(is_divergent, Δ ≥ 0 ? one(Δ) : exp(Δ), 1)
end

is_divergent(::TrajectoryNUTS, x::DivergenceStatistic) = x.divergent

function combine_divergence_statistics(::TrajectoryNUTS,
                                       x::DivergenceStatistic, y::DivergenceStatistic)
    # A divergent subtree make a tree divergent, but acceptance information is kept.
    DivergenceStatistic(x.divergent || y.divergent, x.∑a + y.∑a, x.steps + y.steps)
end

"""
$(SIGNATURES)

Return the acceptance rate (a `Real` between `0` and `1`).
"""
acceptance_rate(x::DivergenceStatistic) = x.∑a / x.steps

###
### turn analysis
###

"Statistics for the identification of turning points. See Betancourt (2017, appendix)."
struct TurnStatistic{T}
    p♯₋::T
    p♯₊::T
    ρ::T
end

function combine_turn_statistics(::TrajectoryNUTS, x::TurnStatistic, y::TurnStatistic)
    TurnStatistic(x.p♯₋, y.p♯₊, x.ρ + y.ρ)
end

function is_turning(::TrajectoryNUTS, τ::TurnStatistic)
    # Uses the generalized NUTS criterion from Betancourt (2017).
    @unpack p♯₋, p♯₊, ρ = τ
    @argcheck p♯₋ ≢ p♯₊ "internal error: is_turning called on a leaf"
    dot(p♯₋, ρ) < 0 || dot(p♯₊, ρ) < 0
end

###
### leafs
###

function leaf(trajectory::TrajectoryNUTS, z, is_initial)
    @unpack H, π₀, min_Δ = trajectory
    Δ = is_initial ? zero(π₀) : logdensity(H, z) - π₀
    isdiv = Δ < min_Δ
    d = is_initial ? divergence_statistic() : divergence_statistic(isdiv, Δ)
    ζ = isdiv ? nothing : z
    τ = isdiv ? nothing : (p♯ = calculate_p♯(trajectory.H, z); TurnStatistic(p♯, p♯, z.p))
    ζ, Δ, τ, d
end

####
#### NUTS interface
####

"Default maximum depth for trees."
const MAX_DEPTH = 10

"""
$(TYPEDEF)

Options for building NUTS trees. These are the parameters that are expected to remain stable
during adaptation and sampling.
"""
struct TreeOptionsNUTS
    max_depth::Int
    min_Δ::Float64
    function TreeOptionsNUTS(; max_depth = MAX_DEPTH, min_Δ = -1000.0)
        @argcheck 0 < max_depth ≤ MAX_DIRECTIONS_DEPTH
        @argcheck min_Δ < 0
        new(Int(max_depth), Float64(min_Δ))
    end
end

"""
$(TYPEDEF)

Diagnostic information for a single tree built with the No-U-turn sampler.

Accessing fields directly is part of the API.
"""
struct TreeStatisticsNUTS
    "Log density (negative energy)."
    π::Float64
    "Depth of the tree."
    depth::Int
    "Reason for termination."
    termination::Termination
    "Average acceptance probability."
    acceptance_statistic::Float64
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
                          Q::EvaluatedLogDensity, ϵ)
    z = PhasePoint(Q, rand_p(rng, H.κ))
    trajectory = TrajectoryNUTS(H, logdensity(H, z), ϵ, options.min_Δ)
    directions = rand(rng, Directions)
    ζ, d, termination, depth = sample_trajectory(rng, trajectory, z, options.max_depth,
                                                 directions)
    statistics = TreeStatisticsNUTS(logdensity(H, ζ), depth, termination,
                                    acceptance_rate(d), d.steps, directions)
    ζ.Q, statistics
end
