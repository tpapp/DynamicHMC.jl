#####
##### Building blocks for sampling.
#####
##### Only NUTS_Transition and the exported functions are part of the API.
#####

export NUTS_Transition, get_position, get_neg_energy, get_depth, get_termination,
    get_acceptance_rate, get_steps

####
#### Trajectory and implementation
####

"""
Representation of a trajectory, ie a Hamiltonian with a discrete integrator that
also checks for divergence.
"""
struct Trajectory{TH,Tf}
    "Hamiltonian."
    H::TH
    "Log density of z (negative log energy) at initial point."
    π₀::Tf
    "Stepsize for leapfrog."
    ϵ::Tf
    "Smallest decrease allowed in the log density."
    min_Δ::Tf
end

"""
    $(SIGNATURES)

Convenience constructor for trajectory.
"""
Trajectory(H, π₀, ϵ; min_Δ = -1000.0) = Trajectory(H, π₀, ϵ, min_Δ)

function move(trajectory::Trajectory, z, fwd)
    @unpack H, ϵ = trajectory
    leapfrog(H, z, fwd ? ϵ : -ϵ)
end

###
### proposals
###

"""
Proposal that is propagated through by sampling recursively when building the
trees.
"""
struct Proposal{Tz,Tf}
    "Proposed point."
    z::Tz
    "Log weight (log(∑ exp(Δ)) of trajectory/subtree)."
    ω::Tf
end

"""
    logprob, ω = combined_logprob_logweight(ω₁, ω₂, bias)

Given (relative) log probabilities `ω₁` and `ω₂`, return the log probabiliy of
drawing a sampel from the second (`logprob`) and the combined (relative) log
probability (`ω`).

When `bias`, biases towards the second argument, introducing anti-correlations.
"""
function combined_logprob_logweight(ω₁, ω₂, bias)
    ω = logaddexp(ω₁, ω₂)
    ω₂ - (bias ? ω₁ : ω), ω
end

function combine_proposals(rng, ::Trajectory, ζ₁::Proposal, ζ₂::Proposal,
                           is_forward, is_doubling)
    # when doubling, use biased progressive sampling
    logprob, ω = combined_logprob_logweight(ζ₁.ω, ζ₂.ω, is_doubling)
    z = (logprob ≥ 0 || rand_bool(rng, exp(logprob))) ? ζ₂.z : ζ₁.z
    Proposal(z, ω)
end

####
#### divergence statistics
####

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

is_divergent(::Trajectory, x::DivergenceStatistic) = x.divergent

function combine_divergence_statistics(::Trajectory,
                                       x::DivergenceStatistic, y::DivergenceStatistic)
    # A divergent subtree make a tree divergent, but acceptance information is kept.
    DivergenceStatistic(x.divergent || y.divergent, x.∑a + y.∑a, x.steps + y.steps)
end

####
#### turn analysis
####

"Statistics for the identification of turning points. See Betancourt (2017, appendix)."
struct TurnStatistic{T}
    p♯₋::T
    p♯₊::T
    ρ::T
end

function combine_turn_statistics(::Trajectory, x::TurnStatistic, y::TurnStatistic)
    TurnStatistic(x.p♯₋, y.p♯₊, x.ρ + y.ρ)
end

function is_turning(::Trajectory, τ::TurnStatistic)
    # Uses the generalized NUTS criterion from Betancourt (2017).
    @unpack p♯₋, p♯₊, ρ = τ
    dot(p♯₋, ρ) < 0 || dot(p♯₊, ρ) < 0
end

###
### leafs
###

function leaf(trajectory::Trajectory, z, is_initial)
    @unpack H, π₀, min_Δ = trajectory
    Δ = is_initial ? zero(π₀) : neg_energy(H, z) - π₀
    isdiv = min_Δ > Δ
    d = is_initial ? divergence_statistic() : divergence_statistic(isdiv, Δ)
    ζ = isdiv ? nothing : Proposal(z, Δ)
    τ = isdiv ? nothing : (p♯ = get_p♯(trajectory.H, z); TurnStatistic(p♯, p♯, z.p))
    ζ, τ, d
end

####
#### NUTS interface
####

"""
    get_acceptance_rate(x)

Return average Metropolis acceptance rate.
"""
get_acceptance_rate(x::DivergenceStatistic) = x.∑a / x.steps

"""
Single transition by the No-U-turn sampler. Contains new position and
diagnostic information.
"""
struct NUTS_Transition{Tv,Tf}
    "New position."
    q::Tv
    "Log density (negative energy)."
    π::Tf
    "Depth of the tree."
    depth::Int
    "Reason for termination."
    termination::Termination
    "Average acceptance probability."
    a::Tf
    "Number of leapfrog steps evaluated."
    steps::Int
end

"Position after transition."
get_position(x::NUTS_Transition) = x.q

"Negative energy of the Hamiltonian at the position."
get_neg_energy(x::NUTS_Transition) = x.π

"Tree depth."
get_depth(x::NUTS_Transition) = x.depth

"Reason for termination, see [`Termination`](@ref)."
get_termination(x::NUTS_Transition) = x.termination

"Average acceptance rate over trajectory."
get_acceptance_rate(x::NUTS_Transition) = x.a

"Number of integrator steps."
get_steps(x::NUTS_Transition) = x.steps

"""
    NUTS_transition(rng, H, q, ϵ, max_depth; args...)

No-U-turn Hamiltonian Monte Carlo transition, using Hamiltonian `H`, starting at
position `q`, using stepsize `ϵ`. Builds a doubling dynamic tree of maximum
depth `max_depth`. `args` are passed to the `Trajectory` constructor. `rng` is
the random number generator used.
"""
function NUTS_transition(rng, H, q, ϵ, max_depth; args...)
    z = rand_phasepoint(rng, H, q)
    trajectory = Trajectory(H, neg_energy(H, z), ϵ; args...)
    directions = rand(rng, Directions)
    ζ, d, termination, depth = sample_trajectory(rng, trajectory, z, max_depth, directions)
    NUTS_Transition(ζ.z.q, neg_energy(H, ζ.z), depth, termination,
                    get_acceptance_rate(d), d.steps)
end
