#####
##### Sampling: high-level interface and building blocks
#####

export NUTS, mcmc, mcmc_adapting_ϵ, NUTS_init_tune_mcmc, sample_cov, get_position_matrix

####
#### high-level interface: sampler
####


"""
Specification for the No-U-turn algorithm, including the random number
generator, Hamiltonian, the initial position, and various parameters.
"""
struct NUTS{Tv, Tf, TR, TH, Trep <: AbstractReport}
    "Random number generator."
    rng::TR
    "Hamiltonian"
    H::TH
    "position"
    q::Tv
    "stepsize"
    ϵ::Tf
    "maximum depth of the tree"
    max_depth::Int
    "reporting"
    report::Trep
end

function show(io::IO, nuts::NUTS)
    @unpack q, ϵ, max_depth = nuts
    println(io, "NUTS sampler in $(length(q)) dimensions")
    println(io, "  stepsize (ϵ) ≈ $(round(ϵ; sigdigits = 3))")
    println(io, "  maximum depth = $(max_depth)")
    println(io, "  $(nuts.H.κ)")
end

"""
    mcmc(sampler, N)

Run the MCMC `sampler` for `N` iterations, returning the results as a vector,
which has elements that conform to the sampler.
"""
function mcmc(sampler::NUTS{Tv,Tf}, N::Int) where {Tv,Tf}
    @unpack rng, H, q, ϵ, max_depth, report = sampler
    sample = Vector{NUTS_Transition{Tv,Tf}}(undef, N)
    start_progress!(report, "MCMC"; total_count = N)
    for i in 1:N
        trans = NUTS_transition(rng, H, q, ϵ, max_depth)
        q = trans.q
        sample[i] = trans
        report!(report, i)
    end
    end_progress!(report)
    sample
end

"""
    sample, A = mcmc_adapting_ϵ(rng, sampler, N, [A_params, A])

Same as [`mcmc`](@ref), but [`tune`](@ref) stepsize ϵ according to the
parameters `A_params` and initial state `A`. Return the updated `A` as the
second value.

When the last two parameters are not specified, initialize using `adapting_ϵ`.
"""
function mcmc_adapting_ϵ(sampler::NUTS{Tv,Tf}, N::Int, A_params, A) where {Tv,Tf}
    @unpack rng, H, q, max_depth, report = sampler
    sample = Vector{NUTS_Transition{Tv,Tf}}(undef, N)
    start_progress!(report, "MCMC, adapting ϵ"; total_count = N)
    for i in 1:N
        trans = NUTS_transition(rng, H, q, get_current_ϵ(A), max_depth)
        A = adapt_stepsize(A_params, A, trans.a)
        q = trans.q
        sample[i] = trans
        report!(report, i)
    end
    end_progress!(report)
    sample, A
end

mcmc_adapting_ϵ(sampler::NUTS, N) =
    mcmc_adapting_ϵ(sampler, N, adapting_ϵ(sampler.ϵ)...)

"""
    variable_matrix(posterior)

Return the samples of the parameter vector as rows of a matrix.
"""
get_position_matrix(sample) = vcat(get_position.(sample)'...)

####
#### tuning and diagnostics
####

"""
$(SIGNATURES)

Covariance matrix of the sample.
"""
sample_cov(sample) = cov(get_position_matrix(sample); dims = 1)

"Default maximum depth for trees."
const MAX_DEPTH = 10

"""
$(SIGNATURES)

Initialize a NUTS sampler for log density `ℓ` using local information.

# Mandatory arguments

- `rng`: the random number generator

- `ℓ`: the log density function specification

# Keyword arguments

- `q`: initial position. *Default*: random (from IID standard normals).

- `κ`: kinetic energy specification. *Default*: Gaussian with identity matrix.

- `p`: initial momentum. *Default*: random from standard multivariate normal.

- `max_depth`: maximum tree depth. *Default*: `$(MAX_DEPTH)`.

- `ϵ`: initial stepsize, or parameters for finding it (passed on to
  [`find_initial_stepsize`](@ref).
"""
function NUTS_init(rng::AbstractRNG, ℓ;
                   q = randn(rng, dimension(ℓ)),
                   κ = GaussianKE(dimension(ℓ)),
                   p = rand(rng, κ),
                   max_depth = MAX_DEPTH,
                   ϵ = InitialStepsizeSearch(),
                   report = ReportIO())
    H = Hamiltonian(ℓ, κ)
    z = phasepoint_in(H, q, p)
    if !(ϵ isa Float64)
        ϵ = find_initial_stepsize(ϵ, H, z)
    end
    NUTS(rng, H, q, ϵ, max_depth, report)
end

####
#### tuning: abstract interface
####

"""
$(TYPEDEF)

A tuner that adapts the sampler.

All subtypes support `length` which returns the number of steps (*note*: if not
in field `N`, define `length` accordingly), other parameters vary.
"""
abstract type AbstractTuner end

length(tuner::AbstractTuner) = tuner.N

"""
    sampler′ = tune(sampler, tuner)

Given a `sampler` (or similar a parametrization) and a `tuner`, return the
updated sampler state after tuning.
"""
function tune end

####
#### tuning: tuner building blocks
####

"Adapt the integrator stepsize for `N` samples."
struct StepsizeTuner <: AbstractTuner
    N::Int
end

show(io::IO, tuner::StepsizeTuner) =
    print(io, "Stepsize tuner, $(tuner.N) samples")

function tune(sampler::NUTS, tuner::StepsizeTuner)
    @unpack rng, H, max_depth, report = sampler
    sample, A = mcmc_adapting_ϵ(sampler, tuner.N)
    NUTS(rng, H, sample[end].q, get_final_ϵ(A), max_depth, report)
end

"""
Tune the integrator stepsize and covariance. Covariance tuning is from scratch
(no prior information is used), regularized towards the identity matrix.
"""
struct StepsizeCovTuner{Tf} <: AbstractTuner
    "Number of samples."
    N::Int
    """
    Regularization factor for normalizing variance. An estimated covariance
    matrix `Σ` is rescaled by `regularize/sample size`` towards `σ²I`, where
    `σ²` is the median of the diagonal.
    """
    regularize::Tf
end

function show(io::IO, tuner::StepsizeCovTuner)
    @unpack N, regularize = tuner
    print(io, "Stepsize and covariance tuner, $(N) samples, regularization $(regularize)")
end

function tune(sampler::NUTS, tuner::StepsizeCovTuner)
    @unpack regularize, N = tuner
    @unpack rng, H, max_depth, report = sampler
    sample, A = mcmc_adapting_ϵ(sampler, N)
    Σ = sample_cov(sample)
    Σ += (UniformScaling(median(diag(Σ)))-Σ) * regularize/N
    # FIXME: Symmetric + Symmetric above does not preserve Symmetric
    κ = GaussianKE(Symmetric(Σ))
    NUTS(rng, Hamiltonian(H.ℓ, κ), sample[end].q, get_final_ϵ(A), max_depth, report)
end

"Sequence of tuners, applied in the given order."
struct TunerSequence{T} <: AbstractTuner
    tuners::T
end

function show(io::IO, tuner::TunerSequence)
    @unpack tuners = tuner
    print(io, "Sequence of $(length(tuners)) tuners, $(length(tuner)) total samples")
    for t in tuners
        print(io, "\n  ")
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
    tuners = Union{StepsizeTuner, StepsizeCovTuner}[StepsizeTuner(init)]
    for _ in 1:M
        tuners = push!(tuners, StepsizeCovTuner(mid, regularize))
        mid *= 2
    end
    push!(tuners, StepsizeTuner(term))
    TunerSequence((tuners..., ))
end

function tune(sampler, seq::TunerSequence)
    for tuner in seq.tuners
        sampler = tune(sampler, tuner)
    end
    sampler
end

"""
$(SIGNATURES)

Init, tune, and then draw `N` samples from `ℓ` using the NUTS algorithm.

Return the *sample* (a vector of [`NUTS_transition`](@ref)s) and the *tuned
sampler*.

`rng` is the random number generator.

`args` are passed on to various methods, see [`NUTS_init`](@ref) and
[`bracketed_doubling_tuner`](@ref).

Most users would use this function, unless they are doing something that
requires manual tuning.
"""
function NUTS_init_tune_mcmc(rng::AbstractRNG, ℓ, N::Integer; args...)
    sampler_init = NUTS_init(rng, ℓ; args...)
    sampler_tuned = tune(sampler_init, bracketed_doubling_tuner(; args...))
    mcmc(sampler_tuned, N), sampler_tuned
end

"""
$SIGNATURES

Same as the other method, but with random number generator `Random.GLOBAL_RNG`.
"""
NUTS_init_tune_mcmc(ℓ, N::Integer; args...) =
    NUTS_init_tune_mcmc(Random.GLOBAL_RNG, ℓ, N; args...)
