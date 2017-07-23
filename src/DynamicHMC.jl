"""
Notation follows Betancourt (2017), with some differences.

Instead of energies, densities (= - energy) are used in the code.

ℓ: log density (we sample from)
κ: distribution/density that corresponds to kinetic energy 
H: Hamiltonian
q: position
p: momentum
z: point in phase space (q,p)
ϵ: stepsize
a: acceptance rate
A: acceptance tuning state
ζ: proposal from trajectory (phase point and weight)
τ: turn statistic
d: divergence statistic
π: log density (**different from papers**)
Δ: logdensity relative to initial point of trajectory
"""
module DynamicHMC

using ArgCheck
using DataStructures
using Parameters

import Base: rand, length, show
import Base.LinAlg.checksquare
import StatsFuns: logsumexp

######################################################################
# Hamiltonian and leapfrog
######################################################################

export
    # Hamiltonian
    KineticEnergy, EuclideanKE, GaussianKE, logdensity, loggradient, Hamiltonian,
    PhasePoint, phasepoint,
    # stepsize
    find_reasonable_logϵ, adapt, DualAveragingParameters, DualAveragingAdaptation, adapting_ϵ,
    # transition 
    HMCTransition, variable, logdensity, depth, termination, acceptance_rate, steps,
    HMC_transition, HMC_adapting_sample, HMC_sample, variable_matrix,
    # tuning and diagnostics
    sample_cov, EBFMI, TunedNUTS, TunedNUTS_init, tune, TunerStepsize,
    TunerStepsizeCov, TunerSequence, bracketed_doubling_tuner,
    HMCStatistics, HMC_statistics, NUTS_sample, NUTS_tune, NUTS_tune_and_sample

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
struct GaussianKE{T <: AbstractMatrix, S <: AbstractMatrix} <: EuclideanKE
    "M⁻¹"
    Minv::T
    "W such that W*W'=M. Used for generating random draws."
    W::S
    function GaussianKE{T, S}(Minv, W) where {T, S}
        @argcheck checksquare(Minv) == checksquare(W)
        new(Minv, W)
    end
end

GaussianKE(M::T, W::S) where {T,S} = GaussianKE{T,S}(M, W)

GaussianKE(Minv::AbstractMatrix) = GaussianKE(Minv, inv(chol(Minv)))

"""
    logdensity(κ, p, [q])

Return the log density of kinetic energy `κ`, at momentum `p`. Some
kinetic energies (eg Riemannian geometry) will need `q`, too.
"""
logdensity(κ::GaussianKE, p, q = nothing) = -dot(p, κ.Minv * p) / 2

getp♯(κ::GaussianKE, p, q = nothing) = κ.Minv * p

loggradient(κ::GaussianKE, p, q = nothing) = -getp♯(κ, p)

rand(rng, κ::GaussianKE, q = nothing) = κ.W * randn(rng, size(κ.W, 1))

"""
    Hamiltonian(ℓ, κ)

Construct a Hamiltonian from the log density `ℓ`, and the kinetic energy
specification `κ`.
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
phasepoint(H, q, p, ∇ℓq = loggradient(H.ℓ, q), ℓq = logdensity(H.ℓ, q)) =
    PhasePoint(q, p, ∇ℓq, ℓq)

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
function leapfrog{Tℓ, Tκ <: EuclideanKE}(H::Hamiltonian{Tℓ,Tκ}, z::PhasePoint, ϵ)
    @unpack ℓ, κ = H
    @unpack p, q, ∇ℓq = z
    pₘ = p + ϵ/2 * ∇ℓq
    q′ = q - ϵ * loggradient(κ, pₘ)
    ∇ℓq′ = loggradient(ℓ, q′)
    p′ = pₘ + ϵ/2 * ∇ℓq′
    phasepoint(H, q′, p′, ∇ℓq′)
end


######################################################################
# stepsize heuristics and adaptation
######################################################################

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
    find_reasonable_logϵ(H, z; tol, a, ϵ₀, maxiter_bracket, maxiter_bisection)

Let

``z′(ϵ) = leapfrog(H, z, ϵ)``

and

``A(ϵ) = exp(logdensity(H, z′) - logdensity(H, z))``

denote the ratio of densities between a point `z` and another point after one
leapfrog step with stepsize `ϵ`.

Returns an `ϵ` such that `|log(A(ϵ)) - log(a)| ≤ tol`. Uses iterative bracketing
(with gently expanding steps) and rootfinding.

Starts at `ϵ₀`, uses `maxiter` iterations for the bracketing and the
rootfinding, respectively.
"""
function find_reasonable_logϵ(H, z; tol = 0.15, a = 0.75, ϵ₀ = 1.0,
                              maxiter_bracket = MAXITER_BRACKET,
                              maxiter_bisection = MAXITER_BISECTION)
    target = logdensity(H, z) + log(a)
    function residual(logϵ)
        z′ = leapfrog(H, z, exp(logϵ))
        logdensity(H, z′) - target
    end
    bracket_find_zero(residual, log(ϵ₀), log(0.5), 1.1, tol;
                      maxiter_bracket = MAXITER_BRACKET,
                      maxiter_bisection = MAXITER_BISECTION)
end

"""
Parameters for the dual averaging algorithm of Gelman and Hoffman (2014,
Algorithm 6).

To get reasonable defaults, initialize with `DualAveragingParameters(logϵ₀)`.
See [`adapting_ϵ`](@ref) for a joint constructor.
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

"Current state of adaptation for `ϵ`. Use
`DualAverageingAdaptation(logϵ₀)` to get an initial value."
struct DualAveragingAdaptation{T <: AbstractFloat}
    m::Int
    H̄::T
    logϵ::T
    logϵ̄::T
end

"""
    getϵ(A, tuning = true)

When `tuning`, return the stepsize `ϵ` for the next HMC step. Otherwise return
the tuned `ϵ`.
"""
getϵ(A::DualAveragingAdaptation, tuning = true) = exp(tuning ? A.logϵ : A.logϵ̄)

DualAveragingAdaptation(logϵ₀) =
    DualAveragingAdaptation(0, zero(logϵ₀), logϵ₀, zero(logϵ₀))

function adapting_ϵ(ϵ; args...)
    logϵ = log(ϵ)
    DualAveragingParameters(logϵ; args...), DualAveragingAdaptation(logϵ)
end

"""
    A′ = adapt(parameters, A, a)

Update the adaptation `A` of log stepsize `logϵ` with acceptance rate
`a`, using the dual averaging algorithm of Gelman and Hoffman (2014,
Algorithm 6). Return the new adaptation.
"""
function adapt(parameters::DualAveragingParameters, A::DualAveragingAdaptation, a)
    @argcheck 0 ≤ a ≤ 1
    @unpack μ, δ, γ, κ, t₀ = parameters
    @unpack m, H̄, logϵ, logϵ̄ = A
    m += 1
    H̄ += (δ - a - H̄) / (m+t₀)
    logϵ = μ - √m/γ*H̄
    logϵ̄ += m^(-κ)*(logϵ - logϵ̄)
    DualAveragingAdaptation(m, H̄, logϵ, logϵ̄)
end

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
function combine_proposals(rng, ζ₁::Proposal, ζ₂::Proposal, bias)
    logprob, ω = combined_logprob_logweight(ζ₁.ω, ζ₂.ω, bias)
    z = (logprob ≥ 0 || rand_bool(rng, exp(logprob))) ? ζ₂.z : ζ₁.z
    Proposal(z, ω)
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
[`leaf`](@ref).
"""
function isturning(τ::TurnStatistic)
    @unpack p♯₋, p♯₊, ρ = τ
    dot(p♯₋, ρ) < 0 || dot(p♯₊, ρ) < 0
end

######################################################################
# sampling
######################################################################

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

Convenience constructor for trajectory. Uses the leapfrog integrator.
"""
Trajectory(H, π₀, ϵ; min_Δ = -1000.0) = Trajectory(H, π₀, ϵ, min_Δ)

"""
    ζ, τ, d = leaf(trajectory, z, isinitial)

Construct a proposal, turn statistic, and divergence statistic for a single point `z` in
`trajectory`. When `isinitial`, `z` is the initial point in the trajectory.

Return

- `ζ`: the proposal, which should only be used when `!isdivergent(d)`
- `τ`: the turn statistic, which should only be used when `!isdivergent(d)`
- `d`: divergence statistic
"""
function leaf(trajectory::Trajectory, z, isinitial)
    @unpack H, π₀, min_Δ = trajectory
    Δ = isinitial ? zero(π₀) : logdensity(H, z) - π₀
    isdiv = min_Δ > Δ
    d = isinitial ? divergence_statistic() : divergence_statistic(isdiv, Δ)
    ζ = isdiv ? nothing : Proposal(z, Δ)
    τ = isdiv ? nothing : (p♯ = getp♯(trajectory.H, z); TurnStatistic(p♯, p♯, z.p))
    ζ, τ, d
end

function move(trajectory::Trajectory, z, fwd)
    @unpack H, ϵ = trajectory
    leapfrog(H, z, fwd ? ϵ : -ϵ)
end

struct HMCTransition{Tv,Tf}
    "New phasepoint."
    q::Tv
    "Log density."
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
variable(x::HMCTransition) = x.q

"Log density (negative energy of the Hamiltonian) at the position."
logdensity(x::HMCTransition) = x.π

"Tree depth."
depth(x::HMCTransition) = x.depth

"Reason for termination, see [`Termination`](@ref)."
termination(x::HMCTransition) = x.termination

"Average acceptance rate over trajectory."
acceptance_rate(x::HMCTransition) = x.a

"Number of integrator steps."
steps(x::HMCTransition) = x.steps

"""
    HMC_transition(rng, H, q, ϵ, max_depth; args...)

Hamiltonian Monte Carlo transition, using Hamiltonian `H`, starting at position
`q`, using stepsize `ϵ`. Builds a doubling dynamic tree of maximum depth
`max_depth`. `args` are passed to the `Trajectory` constructor. `rng` is the
random number generator used.
"""
function HMC_transition(rng, H, q, ϵ, max_depth; args...)
    z = rand_phasepoint(rng, H, q)
    trajectory = Trajectory(H, logdensity(H, z), ϵ; args...)
    ζ, d, termination, depth = sample_trajectory(rng, trajectory, z, max_depth)
    HMCTransition(ζ.z.q, logdensity(H, ζ.z), depth, termination, acceptance_rate(d), d.steps)
end

function HMC_adapting_sample(rng, H, max_depth::Int, q::Tv, N::Int, DA_params, A) where Tv
    sample = Vector{HMCTransition{Tv, Float64}}(N)
    for i in 1:N
        trans = HMC_transition(rng, H, q, getϵ(A), max_depth)
        A = adapt(DA_params, A, trans.a)
        q = trans.q
        sample[i] .= trans
    end
    sample, A
end

function HMC_sample(rng, H, max_depth::Int, ϵ, q::Tv, N::Int) where Tv
    sample = Vector{HMCTransition{Tv, Float64}}(N)
    for i in 1:N
        trans = HMC_transition(rng, H, q, ϵ, max_depth)
        q = trans.q
        sample[i] .= trans
    end
    sample
end

"""
    variable_matrix(posterior)

Return the samples of the parameter vector as rows of a matrix.
"""
variable_matrix(sample) = vcat(variable.(sample)'...)

######################################################################
# tuning and diagnostics
######################################################################

"""
    sample_cov(sample)

Covariance matrix of the sample.
"""
sample_cov(sample) = cov(variable_matrix(sample), 1)

"""
    EBFMI(sample)

Energy Bayesian fraction of missing information. Useful for diagnosing poorly
chosen kinetic energies.
"""
function EBFMI(sample)
    sum(abs2, diff(logdensity.(sample))) / var(πs)
end

struct TunedNUTS{TH, Tv, Tf}
    "Hamiltonian"
    H::TH
    "position"
    q::Tv
    "stepsize"
    ϵ::Tf
    "maximum depth of the tree"
    max_depth::Int
end

function NUTS_sample(rng, sampler::TunedNUTS, N)
    @unpack H, q, ϵ, max_depth = sampler
    HMC_sample(rng, H, max_depth, ϵ, q, N)
end

"""
    TunedNUTS_init(rng, ℓ, q; Minv = I, logϵ)

Given a density `ℓ` and a position `q`, return an initial TunedNUTS sampler
using local information.
"""
function TunedNUTS_init(rng, ℓ;
                        q = randn(rng, length(ℓ)),
                        Minv = Diagonal(ones(length(ℓ))),
                        max_depth = 5,
                        ϵ = nothing,
                        _...)
    κ = GaussianKE(Minv)
    H = Hamiltonian(ℓ, κ)
    z = rand_phasepoint(rng, H, q)
    if ϵ == nothing
        ϵ = exp(find_reasonable_logϵ(H, z))
    end
    TunedNUTS(H, q, ϵ, max_depth)
end

function HMC_sample_DA(rng, sampler::TunedNUTS, N)
    @unpack H, q, ϵ, max_depth = sampler
    DA_params, A = adapting_ϵ(ϵ)
    HMC_adapting_sample(rng, H, max_depth, q, N, DA_params, A)
end

"Tune the integrator stepsize."
struct TunerStepsize
    N::Int
end

function show(io::IO, tuner::TunerStepsize)
    println(io, "Stepsize tuner, $(tuner.N) samples")
end

length(tuner::TunerStepsize) = tuner.N

"""
    tune(rng, tunestate, tune)

Given a `tunestate` and a `tuner`, return the updated tune state. Use `rng` as a
random number generator.
"""
function tune(rng, sampler::TunedNUTS, tuner::TunerStepsize)
    @unpack H, max_depth = sampler
    sample, A = HMC_sample_DA(rng, sampler, tuner.N)
    TunedNUTS(H, sample[end].q, getϵ(A, false), max_depth)
end

"""
Tune the integrator stepsize and covariance. Covariance tuning is from scratch (no prior information is used), regularized towards the identity matrix.
"""
struct TunerStepsizeCov{Tf}
    "Number of samples."
    N::Int
    """
    Regularization factor for normalizing variance. An estimated covariance matrix `Σ` is rescaled by `regularize/sample size`` towards `σ²I`, where `σ²` is the median of the diagonal.
    """
    regularize::Tf
end

function show(io::IO, tuner::TunerStepsizeCov)
    @unpack N, regularize = tuner
    println(io, "Stepsize and covariance tuner, $(N) samples, regularization $(regularize)")
end

length(tuner::TunerStepsizeCov) = tuner.N

function tune(rng, sampler::TunedNUTS, tuner::TunerStepsizeCov)
    @unpack regularize, N = tuner
    @unpack H, max_depth = sampler
    sample, A = HMC_sample_DA(rng, sampler, N)
    Σ = sample_cov(sample)
    Σ .+= (UniformScaling(median(diag(Σ)))-Σ) * regularize/N
    κ = GaussianKE(Σ)
    TunedNUTS(Hamiltonian(H.ℓ, κ), sample[end].q, getϵ(A), max_depth)
end

"Sequence of tuners, applied in the given order."
struct TunerSequence{T}
    tuners::T
end

function show(io::IO, tuner::TunerSequence)
    @unpack tuners = tuner
    println(io, "Sequence of $(length(tuners)) tuners, $(length(tuner)) total samples")
    for t in tuners
        print(io, "  ")
        show(io, t)
    end
end

length(seq::TunerSequence) = sum(length, seq.tuners)

"""
    bracketed_doubling_tuner(; [init], [mid], [M], [term], [regularize])

A sequence of tuners:

1. tuning stepsize with `init` steps

2. tuning stepsize and covariance: first with `mid` steps, then repeat with
   twice the steps `M` times

3. tuning stepsize with `term` steps

`regularize` is used for covariance regularization.
"""
function bracketed_doubling_tuner(; init = 75, mid = 25, M = 5, term = 50,
                                  regularize = 5.0, _...)
    tunes = Any[TunerStepsize(init)]
    for _ in 1:M
        tunes = push!(tunes, TunerStepsizeCov(mid, regularize))
        mid *= 2
    end
    push!(tunes, TunerStepsize(term))
    TunerSequence(tunes)
end

function tune(rng, sampler, seq::TunerSequence)
    for tuner in seq.tuners
        sampler = tune(rng, sampler, tuner)
    end
    sampler
end

const ACCEPTANCE_QUANTILES = linspace(0,1,5)

"""
    NUTS_tune(rng, ℓ, N; args...)

Given a random number generator `rng` and a log density function `ℓ`, tune the NUTS sampler
"""
function NUTS_tune(rng, ℓ, N; args...)
    init_sampler = TunedNUTS_init(rng, ℓ; args...)
    tune(rng, init_sampler, bracketed_doubling_tuner(; args...))
end

"""
    sample, tuned_sampler = NUTS_tune_and_sample(rng, ℓ, N; args...)

 and then generate `N` samples.

`args` are passed on to [`TunerNUTS_init`](@ref) and [`bracketed_doubling_tuner`](@ref).

`ℓ` should support the following methods:

`logdensity(ℓ, x)`, `loggradient(ℓ, x)`, `length(ℓ)`.

Most users would use this function, unless they are doing something that requires manual tuning.
"""
function NUTS_tune_and_sample(rng, ℓ, N; args...)
    tuned_sampler = NUTS_tune(rng, ℓ, N; args...)
    NUTS_sample(rng, tuned_sampler, N), tuned_sampler
end

struct HMCStatistics{T <: Real,
                     DT <: Associative{Termination,Int},
                     DD <: Associative{Int,Int}}
    "Sample length."
    N::Int
    "average_acceptance"
    a_mean::T
    "acceptance quantiles"
    a_quantiles::Vector{T}
    "termination counts"
    termination_counts::DT
    "depth counts"
    depth_counts::DD
end

"""
    HMC_statistics(sample)

Return statistics about the sample (ie not the variables). Mostly useful for HMC
diagnostics.
"""
function HMC_statistics(sample)
    as = acceptance_rate.(sample)
    HMCStatistics(length(sample),
                  mean(as), quantile(as, ACCEPTANCE_QUANTILES),
                  counter(termination.(sample)), counter(depth.(sample)))
end

function show(io::IO, stats::HMCStatistics)
    @unpack N, a_mean, a_quantiles, termination_counts, depth_counts = stats
    println(io, "Hamiltonian Monte Carlo sample of length $(N)")
    print(io, "  acceptance rate mean: $(round(a_mean,2)), min/25%/median/75%/max:")
    for aq in a_quantiles
        print(io, " ", round(aq,2))
    end
    println(io)
    function print_dict(dict)
        for (key, value) in sort(collect(dict), by = first)
            print(io, " $(key) => $(round(Int, 100*value/N))%")
        end
    end
    print(io, "  termination:")
    print_dict(termination_counts)
    println(io)
    print(io, "  depth:")
    print_dict(depth_counts)
    println(io)
end

end # module
