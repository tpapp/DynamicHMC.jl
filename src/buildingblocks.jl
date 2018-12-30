#####
##### Building blocks for sampling.
#####
##### Only NUTS_Transition and the exported functions are part of the API.
#####

export NUTS_Transition, get_position, get_neg_energy, get_depth, get_termination,
    get_acceptance_rate, get_steps

####
#### utilities
####

"""
$(SIGNATURES)

Random boolean which is `true` with the given probability `prob`.

**All random numbers in this library are obtained from this function.**
"""
rand_bool(rng::AbstractRNG, prob::T) where {T <: AbstractFloat} =
    rand(rng, T) ≤ prob

####
#### abstract trajectory interface
####

"""
    ζ, τ, d, z = adjacent_tree(rng, trajectory, z, depth, fwd)

Traverse the tree of given `depth` adjacent to point `z` in `trajectory`.

`fwd` specifies the direction, `rng` is used for random numbers.

Return:

- `ζ`: the proposal from the tree. Only valid when `!isdivergent(d) &&
  !isturning(τ)`, otherwise the value should not be used.

- `τ`: turn statistics. Only valid when `!isdivergent(d)`.

- `d`: divergence statistics, always valid.

- `z`: the point at the end of the tree.

`trajectory` should support the following interface:

- Starting from leaves: `ζ, τ, d = leaf(trajectory, z, isinitial)`

- Moving along the trajectory: `z = move(trajectory, z, fwd)`

- Testing for turning and divergence: `isturning(τ)`, `isdivergent(d)`

- Combination of return values: `combine_proposals(ζ₁, ζ₂, bias)`,
  `combine_turnstats(τ₁, τ₂)`, and `combine_divstats(d₁, d₂)`
"""
function adjacent_tree(rng, trajectory, z, depth, fwd)
    if depth == 0
        z = move(trajectory, z, fwd)
        ζ, τ, d = leaf(trajectory, z, false)
        ζ, τ, d, z
    else
        ζ₋, τ₋, d₋, z = adjacent_tree(rng, trajectory, z, depth-1, fwd)
        (isdivergent(d₋) || (depth > 1 && isturning(τ₋))) && return ζ₋, τ₋, d₋, z
        ζ₊, τ₊, d₊, z = adjacent_tree(rng, trajectory, z, depth-1, fwd)
        d = combine_divstats(d₋, d₊)
        (isdivergent(d) || (depth > 1 && isturning(τ₊))) && return ζ₊, τ₊, d, z
        τ = fwd ? combine_turnstats(τ₋, τ₊) : combine_turnstats(τ₊, τ₋)
        ζ = isturning(τ) ? nothing : combine_proposals(rng, ζ₋, ζ₊, false)
        ζ, τ, d, z
    end
end

"Reason for terminating a trajectory."
@enum Termination MaxDepth AdjacentDivergent AdjacentTurn DoubledTurn

"""
    ζ, d, termination, depth = sample_trajectory(rng, trajectory, z, max_depth)

Sample a `trajectory` starting at `z`.

Return:

- `ζ`: proposal from the tree

- `d`: divergence statistics

- `termination`: reason for termination (see [`Termination`](@ref))

- `depth`: the depth of the tree that as sampled from. Doubling steps that lead
  to an invalid tree do not contribute to `depth`.

See [`adjacent_tree`](@ref) for the interface that needs to be supported by
`trajectory`.
"""
function sample_trajectory(rng, trajectory, z, max_depth)
    ζ, τ, d = leaf(trajectory, z, true)
    z₋ = z₊ = z
    depth = 0
    termination = MaxDepth
    while depth < max_depth
        fwd = rand_bool(rng, 0.5)
        ζ′, τ′, d′, z = adjacent_tree(rng, trajectory, fwd ? z₊ : z₋, depth, fwd)
        d = combine_divstats(d, d′)
        isdivergent(d) && (termination = AdjacentDivergent; break)
        (depth > 0 && isturning(τ′)) && (termination = AdjacentTurn; break)
        ζ = combine_proposals(rng, ζ, ζ′, true)
        τ = fwd ? combine_turnstats(τ, τ′) : combine_turnstats(τ′, τ)
        fwd ? z₊ = z : z₋ = z
        depth += 1
        isturning(τ) && (termination = DoubledTurn; break)
    end
    ζ, d, termination, depth
end

####
#### proposals
####

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

"""
    combine_proposals(rng, ζ₁, ζ₂, bias)

Combine proposals from two trajectories, using their weights.

When `bias`, biases towards the second proposal, introducing anti-correlations.
"""
function combine_proposals(rng, ζ₁::Proposal, ζ₂::Proposal, bias)
    logprob, ω = combined_logprob_logweight(ζ₁.ω, ζ₂.ω, bias)
    z = (logprob ≥ 0 || rand_bool(rng, exp(logprob))) ? ζ₂.z : ζ₁.z
    Proposal(z, ω)
end

####
#### divergence statistics
####

"""
Divergence and acceptance statistics.

Calculated over all visited phase points (not just the tree that is sampled
from).
"""
struct DivergenceStatistic{Tf}
    "`true` iff the sampler was terminated because of divergence."
    divergent::Bool
    "Sum of metropolis acceptances probabilities over the whole trajectory
    (including invalid parts)."
    ∑a::Tf
    "Total number of leapfrog steps."
    steps::Int
end

"""
    divergence_statistic()

Empty divergence statistic (for initial node).
"""
divergence_statistic() = DivergenceStatistic(false, 0.0, 0)

"""
    divergence_statistic(isdivergent, Δ)

Divergence statistic for leaves. `Δ` is the log density relative to the initial
point.
"""
divergence_statistic(isdivergent, Δ) =
    DivergenceStatistic(isdivergent, Δ ≥ 0 ? one(Δ) : exp(Δ), 1)

"""
    isdivergent(x)

Test if divergence statistic `x` indicates divergence.
"""
isdivergent(x::DivergenceStatistic) = x.divergent

"""
    combine_divstats(x, y)

Combine divergence statistics from (subtrees) `x` and `y`. A divergent subtree
make a subtree divergent.
"""
function combine_divstats(x::DivergenceStatistic, y::DivergenceStatistic)
    DivergenceStatistic(x.divergent || y.divergent,
                        x.∑a + y.∑a, x.steps + y.steps)
end

"""
    get_acceptance_rate(x)

Return average Metropolis acceptance rate.
"""
get_acceptance_rate(x::DivergenceStatistic) = x.∑a / x.steps

####
#### turn analysis
####

"""
Statistics for the identification of turning points. See Betancourt (2017,
appendix).
"""
struct TurnStatistic{T}
    p♯₋::T
    p♯₊::T
    ρ::T
end

"""
    combine_turnstats(x, y)

Combine turn statistics of two trajectories `x` and `y`, which are assume to be
adjacent and in that order.
"""
combine_turnstats(x::TurnStatistic, y::TurnStatistic) =
    TurnStatistic(x.p♯₋, y.p♯₊, x.ρ + y.ρ)

"""
    isturning(τ)

Test termination based on turn statistics. Uses the generalized NUTS criterion
from Betancourt (2017).

Note that this function should not be called with turn statistics returned by
[`leaf`](@ref), ie `depth > 0` is required.
"""
function isturning(τ::TurnStatistic)
    @unpack p♯₋, p♯₊, ρ = τ
    dot(p♯₋, ρ) < 0 || dot(p♯₊, ρ) < 0
end

####
#### sampling
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
    Trajectory(H, π₀, ϵ; min_Δ = -1000.0)

Convenience constructor for trajectory.
"""
Trajectory(H, π₀, ϵ; min_Δ = -1000.0) = Trajectory(H, π₀, ϵ, min_Δ)

"""
    ζ, τ, d = leaf(trajectory, z, isinitial)

Construct a proposal, turn statistic, and divergence statistic for a single
point `z` in `trajectory`. When `isinitial`, `z` is the initial point in the
trajectory.

Return

- `ζ`: the proposal, which should only be used when `!isdivergent(d)`

- `τ`: the turn statistic, which should only be used when `!isdivergent(d)`

- `d`: divergence statistic
"""
function leaf(trajectory::Trajectory, z, isinitial)
    @unpack H, π₀, min_Δ = trajectory
    Δ = isinitial ? zero(π₀) : neg_energy(H, z) - π₀
    isdiv = min_Δ > Δ
    d = isinitial ? divergence_statistic() : divergence_statistic(isdiv, Δ)
    ζ = isdiv ? nothing : Proposal(z, Δ)
    τ = isdiv ? nothing : (p♯ = get_p♯(trajectory.H, z); TurnStatistic(p♯, p♯, z.p))
    ζ, τ, d
end

"""
    move(trajectory, z, fwd)

Return next phase point adjacent to `z` along `trajectory` in the direction
specified by `fwd`.
"""
function move(trajectory::Trajectory, z, fwd)
    @unpack H, ϵ = trajectory
    leapfrog(H, z, fwd ? ϵ : -ϵ)
end

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
    ζ, d, termination, depth = sample_trajectory(rng, trajectory, z, max_depth)
    NUTS_Transition(ζ.z.q, neg_energy(H, ζ.z), depth, termination,
                    get_acceptance_rate(d), d.steps)
end
