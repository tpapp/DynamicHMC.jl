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

"Leapfrog step with given stepsize."
struct Leapfrog{Tf}
    ϵ::Tf
end

"Take a leapfrog step in phase space."
function move{Tℓ, Tκ <: EuclideanKE}(H::Hamiltonian{Tℓ,Tκ}, z::PhasePoint, lf::Leapfrog)
    @unpack ϵ = lf
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

Return `x₁, x₂′, f(x₁), f(x₂′)`. `x₁` and `x₂′ are not necessarily
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
    find_reasonable_logϵ(H, z; tol, a, maxiter, integrator)

Let
``z′(ϵ) = move(H, z, integrator(ϵ))
and
``A(ϵ) = exp(logdensity(H, z′) - logdensity(H, z))``,
denote the ratio of densities between a point `z` and another point
after one leapfrog step with stepsize `ϵ`.

Returns an `ϵ` such that `|log(A(ϵ)) - log(a)| ≤ tol`. Uses iterative
bracketing (with gently expanding steps) and rootfinding.

Starts at `ϵ`, uses `maxiter` iterations for the bracketing and the
rootfinding, respectively.
"""
function find_reasonable_logϵ(H, z, integrator; tol = 0.15, a = 0.75, ϵ = 1.0,
                              maxiter_bracket = MAXITER_BRACKET,
                              maxiter_bisection = MAXITER_BISECTION)
    target = logdensity(H, z) + log(a)
    function residual(logϵ)
        z′ = move(H, z, integrator(exp(logϵ)))
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


###############################################################################
# random booleans
######################################################################

"""
    rand_bool(rng, prob)

Random boolean which is `true` with the given probability `prob`.

This is the only entry point for random numbers in this library.
"""
rand_bool{T <: AbstractFloat}(rng, prob::T) = rand(rng, T) ≤ prob


######################################################################
# abstract trajectory interface
######################################################################

"""
    isterminating(ζ)

Test if a proposal is terminating tree construction.
"""
isterminating(::Void) = true

isvalid(::Any) = true

"""
    ζ, τ, d, z = adjacent_tree(rng, trajectory, z, depth, fwd)

Traverse the tree of given `depth` adjacent to point `z` in `trajectory`.

`fwd` specifies the direction, `rng` is used for random numbers.

Return:

- `ζ`: the proposal from the tree. Only valid when `!isdivergent(d) && !isturning(τ)`,
  otherwise the value should not be used.

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

######################################################################
# proposals
######################################################################

"""
Proposal that is propagated through by sampling recursively when
building the trees.
"""
struct ProposalPoint{Tz,Tf}
    "Proposed point."
    z::Tz
    "Log weight (log(∑ exp(Δ)) of trajectory/subtree)."
    ω::Tf
end

"""
    logprob, ω = combined_logprob_logweight(ω₁, ω₂, bias)

Given (relative) log probabilities `ω₁` and `ω₂`, return the log probabiliy of drawing a
sampel from the second (`logprob`) and the combined (relative) log probability (`ω`).

When `bias`, biases towards the second argument, introducing anti-correlations.
"""
function combined_logprob_logweight(ω₁, ω₂, bias)
    ω = logsumexp(ω₁, ω₂)
    ω₂ - (bias ? ω₁ : ω), ω
end

"""
    combine_proposals(rng, ζ₁, ζ₂, bias)

Combine proposals from two trajectories, using their weights.

When `bias`, biases towards the second proposal, introducing anti-correlations.
"""
function combine_proposals(rng, ζ₁::ProposalPoint, ζ₂::ProposalPoint, bias)
    logprob, ω = combined_logprob_logweight(ζ₁.ω, ζ₂.ω, bias)
    z = (logprob ≥ 0 || rand_bool(rng, exp(logprob))) ? ζ₂.z : ζ₁.z
    ProposalPoint(z, ω)
end

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

"""
    divergence_statistic()

Empty divergence statistic (for initial node).
"""
divergence_statistic() = DivergenceStatistic(false, 0.0, 0)

"""
    divergence_statistic(isdivergent, Δ)

Divergence statistic for leaves. `Δ` is the log density relative to the initial point.
"""
divergence_statistic(isdivergent, Δ) =
    DivergenceStatistic(isdivergent, Δ ≥ 0 ? one(Δ) : exp(Δ), 1)

isdivergent(x::DivergenceStatistic) = x.divergent

function combine_divstats(x::DivergenceStatistic, y::DivergenceStatistic)
    DivergenceStatistic(x.divergent || y.divergent,
                        x.∑a + y.∑a, x.steps + y.steps)
end

acceptance_rate(x::DivergenceStatistic) = x.∑a / x.steps

######################################################################
# turn analysis
######################################################################

"""
Statistics for the identification of turning points. See Betancourt (2017).
"""
struct TurnStatistic{T}
    p♯₋::T
    p♯₊::T
    ρ::T
end

"""
    combine_turnstats(x, y, fwd)

Combine turn statistics of two (adjacent) trajectories `x` and `y`. When `fwd`, `x` is
before `y`, otherwise after.
"""
function combine_turnstats(x::TurnStatistic, y::TurnStatistic, fwd)
    if !fwd
        x, y = y, x
    end
    TurnStatistic(x.p♯₋, y.p♯₊, x.ρ + y.ρ)
end

"""
    isturning(τ)

Test termination based on turn statistics. Uses the generalized NUTS criterion
from Betancourt (2017).

Note that this function should not be called with turn statistics returned by
[`leaf`](@ref).
"""
function isturning(τ::TurnStatistic)
    @unpack p♯₋, p♯₊, ρ = τ
    dot(p♯₋, ρ) < 0 || dot(p♯₊, ρ) < 0
end

######################################################################
# sampling
######################################################################

struct DoublingTrajectory{TH,Tf,Ti}
    "Hamiltonian."
    H::TH
    "Log density of z (negative log energy) at initial point."
    π₀::Tf
    "Integrator (eg leapfrog)."
    integrator::Ti
    "Maximum depth of the binary tree."
    max_depth::Int
    "Smallest decrease allowed in the log density."
    min_Δ::Tf
end

# function DoublingMultinomialSampler(H::TH, π₀::Tf, integrator::Ti; max_depth::Int = 5,
#                            min_Δ::Tf = -1000.0, proposal_type = ProposalPoint) where {Tr, TH, Tf, Tϵ}
#     DoublingMultinomialSampler{Tr, TH, Tf, Tϵ, proposal_type}(rng, H, π₀, ϵ, max_depth, min_Δ)
# end

# proposal_type(::DoublingMultinomialSampler{Tr,TH,Tf,Tϵ,Tp}) where {Tr,TH,Tf,Tϵ,Tp} = Tp

# function Δ_and_divergence(sampler, z)
#     @unpack H, π₀, min_Δ = sampler
#     Δ = logdensity(H, z) - π₀
#     divergent = Δ < min_Δ
#     Δ, DivergenceStatistic(divergent, Δ > 0 ? one(Δ) : exp(Δ), 1)
# end

"""
    ζ, τ, d = leaf(trajectory, z, isinitial)

Construct a proposal, turn statistic, and divergence statistic for a single point `z` in
`trajectory`. When `isinitial`, `z` is the initial point in the trajectory.

Return

- `ζ`: the proposal, which should only be used when `!isdivergent(d)`
- `τ`: the turn statistic, which should only be used when `!isdivergent(d)`
- `d`: divergence statistic
"""
function leaf(trajectory::DoublingTrajectory, z, isinitial)
    Δ = isinitial ? zero(trajectory.π₀) :
        logdensity(trajectory.H, z) - trajectory.π₀
    isdiv = trajectory.min_Δ > Δ
    d = isinitial ? divergence_statistic() : divergence_statistic(isdiv, Δ)
    ζ = isdiv ? nothing : ProposalPoint(z, Δ)
    τ = isdiv ? nothing : (p♯ = getp♯(sampler.H, z); TurnStatistic(p♯, p♯, z.p))
    ζ, τ, d
end

# isvalid(::Void) = false

# isvalid(::Trajectory) = true



# struct HMCTransition{Tv,Tf}
#     "New phasepoint."
#     z::PhasePoint{Tv,Tf}
#     "Depth of the tree."
#     depth::Int
#     "Reason for termination."
#     termination::SamplerTermination
#     "Average acceptance probability."
#     a::Tf
#     "Number of leapfrog steps evaluated."
#     steps::Int
# end

# function HMC_transition(rng, H, z, ϵ; args...)
#     sampler = DoublingMultinomialSampler(rng, H, logdensity(H, z), ϵ; args...)
#     t, d, termination, depth = sample_trajectory(sampler, z)
#     HMCTransition(t.proposal.z, depth, termination, acceptance_rate(d), d.steps)
# end

# HMC_transition(H, z, ϵ; args...) = HMC_transition(GLOBAL_RNG, H, z, ϵ; args...)

# function HMC_sample(rng, H, q::Tv, N, DA_params, A) where Tv
#     posterior = Vector{HMCTransition{Tv, Float64}}(N)
#     for i in 1:N
#         z = rand_phasepoint(rng, H, q)
#         trans = HMC_transition(H, z, exp(A.logϵ))
#         A = adapt(DA_params, A, trans.a)
#         q = trans.z.q
#         posterior[i] .= trans
#     end
#     posterior, A
# end

end # module
