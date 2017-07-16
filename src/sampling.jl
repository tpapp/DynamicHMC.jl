export HMC_transition, HMCTransition, HMC_sample

"""
Representation of a trajectory (and a proposal).

A *trajectory* is a contiguous set of points. A *proposal* is a point
that was selected from this trajectory using multinomal sampling.

Some subtypes *may not have a valid proposal* (because of termination,
divergence, etc). These are considered `invalid` trajectories, and the
only information represented is the reason for that.
"""
struct Trajectory{Tp,Ts}
    "Proposed parameter and its weight."
    proposal::Tp
    "Turn statistics."
    turnstat::Ts
end


struct DoublingMultinomialSampler{Tr,TH,Tf,Tϵ,Tp}
    "Random number generator."
    rng::Tr
    "Hamiltonian."
    H::TH
    "Log density of z (negative log energy) at initial point."
    π₀::Tf
    "Stepsize for leapfrog  integrator (not necessarily a number)."
    ϵ::Tϵ
    "Maximum depth of the binary tree."
    max_depth::Int
    "Smallest decrease allowed in the log density."
    min_Δ::Tf
end

function DoublingMultinomialSampler(rng::Tr, H::TH, π₀::Tf, ϵ::Tϵ; max_depth::Int = 5,
                           min_Δ::Tf = -1000.0, proposal_type = ProposalPoint) where {Tr, TH, Tf, Tϵ}
    DoublingMultinomialSampler{Tr, TH, Tf, Tϵ, proposal_type}(rng, H, π₀, ϵ, max_depth, min_Δ)
end

proposal_type(::DoublingMultinomialSampler{Tr,TH,Tf,Tϵ,Tp}) where {Tr,TH,Tf,Tϵ,Tp} = Tp

function Δ_and_divergence(sampler, z)
    @unpack H, π₀, min_Δ = sampler
    Δ = logdensity(H, z) - π₀
    divergent = Δ < min_Δ
    Δ, DivergenceStatistic(divergent, Δ > 0 ? one(Δ) : exp(Δ), 1)
end

function leaf(sampler, z, Δ)
    p♯ = getp♯(sampler.H, z)
    Trajectory(leaf_proposal(proposal_type(sampler), z, Δ),
               TurnStatistic(p♯, p♯, z.p))
end

isvalid(::Void) = false

isvalid(::Trajectory) = true

"""
FIXME    Nullable(t), z′ = adjacent_tree(sampler, z, depth, ϵ)

Return a tree `t` of given `depth` adjacent to point `z`, created
using `sampler`, with stepsize `ϵ`. The tree `t` is wrapped in a
`Nullable`, to indicate trees we cannot sample from because it would
violate detailed balance (termination, divergence).

`sampler` is the only argument which is modified, recording statistics
for tuning ϵ, and divergence information.

`z′` is returned to mark the end of the tree.
"""
function adjacent_tree(sampler, z, depth, fwd)
    @unpack rng, H, ϵ = sampler
    if depth == 0
        z = leapfrog(H, z, fwd ? ϵ : -ϵ)
        Δ, d = Δ_and_divergence(sampler, z)
        isdivergent(d) ? nothing : leaf(sampler, z, Δ), d, z
    else
        t₋, d₋, z = adjacent_tree(sampler, z, depth-1, fwd)
        isvalid(t₋) || return t₋, d₋, z
        t₊, d₊, z = adjacent_tree(sampler, z, depth-1, fwd)
        d = d₋ ⊔ d₊
        isvalid(t₊) || return t₊, d, z
        if !fwd
            t₋, t₊ = t₋, t₊
        end
        turnstat = t₋.turnstat ⊔ t₊.turnstat
        isturning(turnstat) ? nothing :
            Trajectory(combine_proposals(rng, t₋.proposal, t₊.proposal, false),
                       turnstat), d, z
    end
end


"""
    sample_trajectory(sampler, z, ϵ)
"""
function sample_trajectory(sampler, z)
    @unpack max_depth, π₀, rng = sampler
    t = leaf(sampler, z, zero(π₀))
    d = DivergenceStatistic()
    z₋ = z₊ = z
    depth = 0
    termination = MaxDepth
    while depth < max_depth
        fwd = rand_bool(rng, 0.5)
        t′, d′, z = adjacent_tree(sampler, fwd ? z₊ : z₋, depth, fwd)
        d = d ⊔ d′
        isdivergent(d) && (termination = AdjacentDivergent; break)
        isvalid(t′) || (termination = AdjacentTurn; break)
        proposal = combine_proposals(rng, t.proposal, t′.proposal, true)
        t = Trajectory(proposal, t.turnstat ⊔ t′.turnstat)
        fwd ? z₊ = z : z₋ = z
        depth += 1
        isturning(t.turnstat) && (termination = DoubledTurn; break)
    end
    t, d, termination, depth
end

@enum SamplerTermination MaxDepth AdjacentDivergent AdjacentTurn DoubledTurn

struct HMCTransition{Tv,Tf}
    "New phasepoint."
    z::PhasePoint{Tv,Tf}
    "Depth of the tree."
    depth::Int
    "Reason for termination."
    termination::SamplerTermination
    "Average acceptance probability."
    a::Tf
    "Number of leapfrog steps evaluated."
    steps::Int
end

function HMC_transition(rng, H, z, ϵ; args...)
    sampler = DoublingMultinomialSampler(rng, H, logdensity(H, z), ϵ; args...)
    t, d, termination, depth = sample_trajectory(sampler, z)
    HMCTransition(t.proposal.z, depth, termination, acceptance_rate(d), d.steps)
end

HMC_transition(H, z, ϵ; args...) = HMC_transition(GLOBAL_RNG, H, z, ϵ; args...)

function HMC_sample(rng, H, q::Tv, N, DA_params, A) where Tv
    posterior = Vector{HMCTransition{Tv, Float64}}(N)
    for i in 1:N
        z = rand_phasepoint(rng, H, q)
        trans = HMC_transition(H, z, exp(A.logϵ))
        A = adapt(DA_params, A, trans.a)
        q = trans.z.q
        posterior[i] .= trans
    end
    posterior, A
end
