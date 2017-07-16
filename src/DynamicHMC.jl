"""
Notation follows Betancourt (2017), with some differences.

"""
module DynamicHMC

using ArgCheck
using Parameters
using PDMats

import Base: rand
import StatsFuns: logsumexp

######################################################################
# Hamiltonian and leapfrog
######################################################################

export
    logdensity, loggradient,
    Hamiltonian,
    KineticEnergy, EuclideanKE, GaussianKE,
    PhasePoint, phasepoint,
    HMCTransition, HMC_transition, HMC_sample

"""
Kinetic energy specifications.

For all subtypes, it is assumed that kinetic energy is symmetric in
the momentum `p`, ie.

```julia
logdensity(::KineticEnergy, p, q) == logdensity(::KineticEnergy, -p, q)
```

When the above is violated, various implicit assumptions will not hold.
"""
abstract type KineticEnergy end

"Euclidean kinetic energies (position independent)."
abstract type EuclideanKE <: KineticEnergy end

"""
Gaussian kinetic energy.

p | q ∼ Normal(0, M)     (importantly, independent of q)

The inverse M⁻¹ is stored.
"""
struct GaussianKE{T <: AbstractPDMat} <: EuclideanKE
    "M⁻¹"
    Minv::T
end

"""
    logdensity(κ, p[, q])

Return the log density of kinetic energy `κ`, at momentum `p`. Some
kinetic energies (eg Riemannian geometry)  will need `q`, too.
"""
logdensity(κ::GaussianKE, p, q = nothing) = -quad(κ.Minv, p) / 2

getp♯(κ::GaussianKE, p, q = nothing) = κ.Minv * p

loggradient(κ::GaussianKE, p, q = nothing) = -getp♯(κ, p)

rand(rng, κ::GaussianKE, q = nothing) = whiten(κ.Minv, randn(rng, dim(κ.Minv)))

"""
    Hamiltonian(ℓ, κ)

Construct a Hamiltonian from the posterior density `ℓ`, and the
kinetic energy specification `κ`.
"""
struct Hamiltonian{Tℓ, Tκ}
    "The (log) density we are sampling from."
    ℓ::Tℓ
    "The kinetic energy."
    κ::Tκ
end

"""
A point in phase space, consists of a position and a momentum.

Log densities and gradients may be saved for speed gains, so a
`PhasePoint` should only be used with a specific Hamiltonian.
"""
struct PhasePoint{Tv,Tf}
    "Position."
    q::Tv
    "Momentum."
    p::Tv
    "Gradient of ℓ at q. Cached for reuse in leapfrog."
    ∇ℓq::Tv
    "ℓ at q. Cached for reuse in sampling."
    ℓq::Tf
end

"""
    phasepoint(H, q, p)

Preferred constructor for phasepoints, computes cached information.
"""
phasepoint(H, q, p) = PhasePoint(q, p, loggradient(H.ℓ, q), logdensity(H.ℓ, q))

"""
    rand_phasepoint(rng, H, q)

Extend a position `q` to a phasepoint with a random momentum according
to the kinetic energy of `H`.
"""
rand_phasepoint(rng, H, q) = phasepoint(H, q, rand(rng, H.κ))
    
"""
Log density for Hamiltonian `H` at point `z`.
"""
logdensity(H::Hamiltonian, z::PhasePoint) = z.ℓq + logdensity(H.κ, z.p, z.q)

getp♯(H::Hamiltonian, z::PhasePoint) = getp♯(H.κ, z.p, z.q)

"Take a leapfrog step in phase space."
function leapfrog{Tℓ, Tκ <: EuclideanKE}(H::Hamiltonian{Tℓ,Tκ}, z::PhasePoint, ϵ)
    @unpack ℓ, κ = H
    @unpack p, q, ∇ℓq = z
    pₘ = p + ϵ/2 * ∇ℓq
    q′ = q - ϵ * loggradient(κ, pₘ)
    ∇ℓq′ = loggradient(ℓ, q′)
    p′ = pₘ + ϵ/2 * ∇ℓq′
    PhasePoint(q′, p′, ∇ℓq′, logdensity(ℓ, q))
end

######################################################################
# stepsize heuristics and adaptation
######################################################################

export
    find_reasonable_logϵ,
    adapt,
    DualAveragingParameters, DualAveragingAdaptation, adapting_logϵ,
    FixedStepSize, fixed_logϵ

const MAXITER_BRACKET = 50
const MAXITER_BISECTION = 50

"""
    bracket_zero(f, x, Δ, C; maxiter)

Find `x₁`, `x₂′` that bracket `f(x) = 0`. `f` should be monotone, use
`Δ > 0` for increasing and `Δ < 0` decreasing `f`.

Return `x₁, x₂′, f(x₁), f(x₂)′`. `x₁` and `x₂′ are not necessarily
ordered.

Algorithm: start at the given `x`, adjust by `Δ` — for increasing `f`,
use `Δ > 0`. At each step, multiply `Δ` by `C`. Stop and throw an
error after `maxiter` iterations.
"""
function bracket_zero(f, x, Δ, C; maxiter = MAXITER_BRACKET)
    @argcheck C > 1
    @argcheck Δ ≠ 0
    fx = f(x)
    s = sign(fx)
    for _ in 1:maxiter
        x′ = x - s*Δ            # note: in the unlikely case s=0 ...
        fx′ = f(x′)
        if s*fx′ ≤ 0            # ... should still work
            return x, fx, x′, fx′
        else
            if abs(fx′) > abs(fx)
                warn("Residual increased, function may not be monotone.")
            end
            Δ *= C
            x = x′
            fx = fx′
        end
    end
    error("Reached maximum number of iterations without crossing 0.")
end

"""
    find_zero(f, a, b, tol; fa, fb, maxiter)

Use bisection to find ``x ∈ [a,b]`` such that `|f(x)| < tol`. When `f`
is costly, specify `fa` and `fb`.

When does not converge within `maxiter` iterations, throw an error.
"""
function find_zero(f, a, b, tol; fa=f(a), fb=f(b), maxiter = MAXITER_BISECTION)
    @argcheck fa*fb ≤ 0 "Initial values don't bracket the root."
    @argcheck tol > 0
    for _ in 1:maxiter
        x = middle(a,b)
        fx = f(x)
        if abs(fx) ≤ tol
            return x
        elseif fx*fa > 0
            a = x
            fa = fx
        else
            b = x
            fb = fx
        end
    end
    error("Reached maximum number of iterations.")
end

function bracket_find_zero(f, x, Δ, C, tol;
                           maxiter_bracket = MAXITER_BRACKET,
                           maxiter_bisection = MAXITER_BISECTION)
    a, fa, b, fb = bracket_zero(f, x, Δ, C; maxiter = maxiter_bracket)
    find_zero(f, a, b, tol; fa=fa, fb = fb, maxiter = maxiter_bisection)
end

"""
    find_reasonable_logϵ(H, z; tol, a, ϵ, maxiter)

Let

``A(ϵ) = exp(logdensity(H, leapfrog(H, z, ϵ)) - logdensity(H, z))``,

denote the ratio of densities between a point `z` and another point
after one leapfrog step with stepsize `ϵ`.

Returns an `ϵ` such that `|log(A(ϵ)) - log(a)| ≤ tol`. Uses iterative
bracketing (with gently expanding steps) and rootfinding.

Starts at `ϵ`, uses `maxiter` iterations for the bracketing and the
rootfinding, respectively.
"""
function find_reasonable_logϵ(H, z; tol = 0.15, a = 0.75, ϵ = 1.0,
                              maxiter_bracket = MAXITER_BRACKET,
                              maxiter_bisection = MAXITER_BISECTION)
    target = logdensity(H, z) + log(a)
    function residual(logϵ)
        z′ = leapfrog(H, z, exp(logϵ))
        logdensity(H, z′) - target
    end
    bracket_find_zero(residual, log(ϵ), log(0.5), 1.1, tol;
                      maxiter_bracket = MAXITER_BRACKET,
                      maxiter_bisection = MAXITER_BISECTION)
end

"""
Parameters for the dual averaging algorithm of Gelman and Hoffman
(2014, Algorithm 6).

To get reasonable defaults, initialize with
`DualAveragingParameters(logϵ₀)`.
"""
struct DualAveragingParameters{T}
    μ::T
    δ::T
    γ::T
    κ::T
    t₀::Int
end

DualAveragingParameters(logϵ₀; δ = 0.65, γ = 0.05, κ = 0.75, t₀ = 10) =
    DualAveragingParameters(promote(log(10)+logϵ₀, δ, γ, κ)..., t₀)

"Current state of adaptation for `ϵ`. Use
`DualAverageingAdaptation(logϵ₀)` to get an initial value."
struct DualAveragingAdaptation{T <: AbstractFloat}
    m::Int
    H̄::T
    logϵ::T
    logϵ̄::T
end

DualAveragingAdaptation(logϵ₀) =
    DualAveragingAdaptation(0, zero(logϵ₀), logϵ₀, zero(logϵ₀))

function adapting_logϵ(logϵ; args...)
    DualAveragingParameters(logϵ; args...), DualAveragingAdaptation(logϵ)
end

"""
    A′ = adapt(parameters, A, a)

Update the adaptation `A` of log stepsize `logϵ` with acceptance rate
`a`, using the dual averaging algorithm of Gelman and Hoffman (2014,
Algorithm 6). Return the new adaptation.
"""
function adapt(parameters::DualAveragingParameters, A::DualAveragingAdaptation, a)
    @unpack μ, δ, γ, κ, t₀ = parameters
    @unpack m, H̄, logϵ, logϵ̄ = A
    m += 1
    H̄ += (δ - a - H̄) / (m+t₀)
    logϵ = μ - √m/γ*H̄
    logϵ̄ += m^(-κ)*(logϵ - logϵ̄)
    DualAveragingAdaptation(m, H̄, logϵ, logϵ̄)
end

struct FixedStepSize{T}
    logϵ::T
end

adapt(::Any, A::FixedStepSize, a) = A

fixed_logϵ(logϵ) = nothing, FixedStepSize(logϵ)

######################################################################
# divergence statistics
######################################################################

struct DivergenceStatistic{Tf}
    "`true` iff the sampler was terminated because of divergence."
    divergent::Bool
    "Sum of metropolis acceptances probabilities over the whole
    trajectory (including invalid parts)."
    ∑a::Tf
    "Total number of leapfrog steps."
    steps::Int
end

DivergenceStatistic() = DivergenceStatistic(false, 0.0, 0)

isdivergent(x::DivergenceStatistic) = x.divergent

function ⊔(x::DivergenceStatistic, y::DivergenceStatistic)
    DivergenceStatistic(x.divergent || y.divergent, x.∑a + y.∑a, x.steps + y.steps)
end

acceptance_rate(x::DivergenceStatistic) = x.∑a / x.steps

###############################################################################
# proposal
######################################################################

"""
    rand_bool(rng, prob)

Random boolean which is `true` with the given`probability `prob`.
"""
rand_bool{T <: AbstractFloat}(rng::AbstractRNG, prob::T) = rand(rng, T) ≤ prob

"""
Proposal that is propagated through by sampling recursively when
building the trees.
"""
struct ProposalPoint{Tz,Tf}
    "Proposed point."
    z::Tz
    "Log weight (log(∑ exp(Δ)) of trajectory/subtree)."
    logweight::Tf
end

function transition_logprob_logweight(logweight_x, logweight_y, bias_y)
    logweight = logsumexp(logweight_x, logweight_y)
    logweight_y - (bias_y ? logweight_x : logweight), logweight
end

"""
    combine_proposal(x, y, bias)

Combine proposals from two trajectories, using their weights.

When `bias_y`, biases towards `y`, introducing anti-correlations.
"""
function combine_proposals(rng, x::ProposalPoint, y::ProposalPoint, bias_y)
    logprob_y, logweight = transition_logprob_logweight(x.logweight, y.logweight, bias_y)
    z = (logprob_y ≥ 0 || rand_bool(rng, exp(logprob_y))) ? y.z : x.z
    ProposalPoint(z, logweight)
end

leaf_proposal(::Type{ProposalPoint}, z, Δ) = ProposalPoint(z, Δ)

######################################################################
# turn analysis
######################################################################

"""
Statistics for the identification of turning points. See Betancourt (2017).

Note that the ± subscripts of p♯ indicate direction of expansion.
"""
struct TurnStatistic{T}
    p♯₋::T
    p♯₊::T
    ρ::T
end

"""
Combine turn statistics of two (adjacent) trajectories.
"""
⊔(x::TurnStatistic, y::TurnStatistic) = TurnStatistic(x.p♯₋, y.p♯₊, x.ρ + y.ρ)

"""
Test termination based on turn statistics. Uses the generalized NUTS
criterion from Betancourt (2017).
"""
function isturning(stats::TurnStatistic)
    @unpack p♯₋, p♯₊, ρ = stats
    dot(p♯₋, ρ) < 0 || dot(p♯₊, ρ) < 0
end

######################################################################
# sampling
######################################################################

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

end # module
