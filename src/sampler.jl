export
    NUTS, mcmc, mcmc_adapting_ϵ, NUTS_init_tune_mcmc, sample_cov,
    get_position_matrix

# high-level interface: sampler

"""
Specification for the No-U-turn algorithm, including the random number
generator, Hamiltonian, the initial position, and various parameters.
"""
struct NUTS{Tv, Tf, TR, TH}
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
end

function show(io::IO, nuts::NUTS)
    @unpack q, ϵ, max_depth = nuts
    println(io, "NUTS sampler in $(length(q)) dimensions")
    println(io, "  stepsize (ϵ) ≈ $(signif(ϵ, 3))")
    println(io, "  maximum depth = $(max_depth)")
    println(io, "  $(nuts.H.κ)")
end

"""
    mcmc(sampler, N)

Run the MCMC `sampler` for `N` iterations, returning the results as a vector,
which has elements that conform to the sampler.
"""
function mcmc(sampler::NUTS{Tv,Tf}, N::Int) where {Tv,Tf}
    @unpack rng, H, q, ϵ, max_depth = sampler
    sample = Vector{NUTS_Transition{Tv,Tf}}(N)
    for i in 1:N
        trans = NUTS_transition(rng, H, q, ϵ, max_depth)
        q = trans.q
        sample[i] .= trans
    end
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
    @unpack rng, H, q, max_depth = sampler
    sample = Vector{NUTS_Transition{Tv,Tf}}(N)
    for i in 1:N
        trans = NUTS_transition(rng, H, q, get_ϵ(A), max_depth)
        A = adapt_stepsize(A_params, A, trans.a)
        q = trans.q
        sample[i] .= trans
    end
    sample, A
end

mcmc_adapting_ϵ(sampler::NUTS, N) =
    mcmc_adapting_ϵ(sampler, N, adapting_ϵ(sampler.ϵ)...)

"""
    variable_matrix(posterior)

Return the samples of the parameter vector as rows of a matrix.
"""
get_position_matrix(sample) = vcat(get_position.(sample)'...)


# tuning and diagnostics


"""
    sample_cov(sample)

Covariance matrix of the sample.
"""
sample_cov(sample) = cov(get_position_matrix(sample), 1)

"""
    NUTS_init(rng, ℓ, q; κ = GaussianKE(length(q)), p, max_depth, ϵ)

Initialize a NUTS sampler for log density `ℓ` using local information.

# Arguments

- `rng`: the random number generator

- `ℓ`: the likelihood function, should return a type that supports
  `DiffResults.value` and `DiffResults.gradient`

- `q`: initial position.

- `κ`: kinetic energy specification. *Default*: Gaussian with identity matrix.

- `p`: initial momentum. *Default*: random from standard multivariate normal.

- `max_depth`: maximum tree depth. *Default*: `5`.

- `ϵ`: initial stepsize, or parameters for finding it (passed on to
  [`find_initial_stepsize`](@ref).
"""
function NUTS_init(rng, ℓ, q;
                   κ = GaussianKE(length(q)),
                   p = rand(rng, κ),
                   max_depth = 5,
                   ϵ = InitialStepsizeSearch())
    H = Hamiltonian(ℓ, κ)
    z = phasepoint_in(H, q, p)
    if !(ϵ isa Float64)
        ϵ = find_initial_stepsize(ϵ, H, z)
    end
    NUTS(rng, H, q, ϵ, max_depth)
end

"""
    NUTS_init(rng, ℓ, dim::Integer; args...)

Random initialization with position `randn(dim)`, all other arguments are passed
on the the other method of this function.
"""
NUTS_init(rng, ℓ, dim::Integer; args...) = NUTS_init(rng, ℓ, randn(dim); args...)


# tuning: abstract interface

"""
A tuner that adapts the sampler.

All subtypes support `length` which returns the number of steps (*note*: if not
in field `N`, define `length` accordingly), other parameters vary.
"""
abstract type AbstractTuner end

length(tuner::AbstractTuner) = tuner.N

"""
    sampler′ = tune(sampler, tune)

Given a `sampler` (or similar a parametrization) and a `tuner`, return the
updated sampler state after tuning.
"""
function tune end


# tuning: tuner building blocks

"Adapt the integrator stepsize for `N` samples."
struct StepsizeTuner <: AbstractTuner
    N::Int
end

show(io::IO, tuner::StepsizeTuner) =
    print(io, "Stepsize tuner, $(tuner.N) samples")

function tune(sampler::NUTS, tuner::StepsizeTuner)
    @unpack rng, H, max_depth = sampler
    sample, A = mcmc_adapting_ϵ(sampler, tuner.N)
    NUTS(rng, H, sample[end].q, get_ϵ(A, false), max_depth)
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
    @unpack rng, H, max_depth = sampler
    sample, A = mcmc_adapting_ϵ(sampler, N)
    Σ = sample_cov(sample)
    Σ .+= (UniformScaling(median(diag(Σ)))-Σ) * regularize/N
    κ = GaussianKE(Σ)
    NUTS(rng, Hamiltonian(H.ℓ, κ), sample[end].q, get_ϵ(A), max_depth)
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
    TunerSequence((tuners...))
end

function tune(sampler, seq::TunerSequence)
    for tuner in seq.tuners
        sampler = tune(sampler, tuner)
    end
    sampler
end

"""
    $SIGNATURES

Init, tune, and then draw `N` samples from `ℓ` using the NUTS algorithm.

Return the *sample* (a vector of [`NUTS_transition`](@ref)s) and the *tuned
sampler*.

`rng` is the random number generator.

`q_or_dim` is a starting position or the dimension (for random initialization).

`args` are passed on to various methods, see [`NUTS_init`](@ref) and
[`bracketed_doubling_tuner`](@ref).

For parameters `q`, `ℓ(q)` should return an object that support the following
methods: `DiffResults.value`, `DiffResults.gradient`.

Most users would use this function, unless they are doing something that
requires manual tuning.
"""
function NUTS_init_tune_mcmc(rng, ℓ, q_or_dim, N::Int; args...)
    sampler_init = NUTS_init(rng, ℓ, q_or_dim; args...)
    sampler_tuned = tune(sampler_init, bracketed_doubling_tuner(; args...))
    mcmc(sampler_tuned, N), sampler_tuned
end

"""
    $SIGNATURES

Same as the other method, but with random number generator
`Base.Random.GLOBAL_RNG`.
"""
NUTS_init_tune_mcmc(ℓ, q_or_dim, N::Int; args...) =
    NUTS_init_tune_mcmc(Base.Random.GLOBAL_RNG, ℓ, q_or_dim, N; args...)
