#####
##### Sampling: high-level interface and building blocks
#####

"Significant digits to display for reporting."
const REPORT_SIGDIGITS = 3

####
#### parts unaffected by warmup
####

"""
$(TYPEDEF)

A log density bundled with an RNG and options for sampling. Contains the parts of the
problem which are not changed during warmup.

# Fields

$(FIELDS)
"""
struct SamplingLogDensity{R,L,O,S}
    "Random number generator."
    rng::R
    "Log density."
    ℓ::L
    """
    Algorithm used for sampling, also contains the relevant parameters that are not affected
    by adaptation. See eg [`NUTS`](@ref).
    """
    algorithm::O
    "Reporting warmup information and chain progress."
    reporter::S
end

####
#### warmup building blocks
####

###
### warmup state
###

"""
$(TYPEDEF)

Representation of a warmup state. Not part of the API.

# Fields

$(FIELDS)
"""
struct WarmupState{TQ <: EvaluatedLogDensity,Tκ <: KineticEnergy, Tϵ <: Union{Real,Nothing}}
    Q::TQ
    κ::Tκ
    ϵ::Tϵ
end

function Base.show(io::IO, warmup_state::WarmupState)
    @unpack κ, ϵ = warmup_state
    ϵ_display = ϵ ≡ nothing ? "unspecified" : "≈ $(round(ϵ; sigdigits = REPORT_SIGDIGITS))"
    print(io, "adapted sampling parameters: stepsize (ϵ) $(ϵ_display), $(κ)")
end

###
### warmup interface and stages
###

"""
$(SIGNATURES)

Return the *results* and the *next warmup state* after warming up/adapting according to
`warmup_stage`, starting from `warmup_state`.

Use `nothing` for a no-op.
"""
function warmup(sampling_logdensity::SamplingLogDensity, warmup_stage::Nothing, warmup_state)
    nothing, warmup_state
end

"""
$(SIGNATURES)

Helper function to create random starting positions in the `[-2,2]ⁿ` box.
"""
random_position(rng, N) = rand(rng, N) .* 4 .- 2

"Docstring for initial warmup arguments."
const DOC_INITIAL_WARMUP_ARGS =
"""
- `q`: initial position. *Default*: random (uniform [-2,2] for each coordinate).

- `κ`: kinetic energy specification. *Default*: Gaussian with identity matrix.

- `ϵ`: a scalar for initial stepsize, or `nothing` for heuristic finders.
"""

"""
$(SIGNATURES)

Create an initial warmup state from a random position.

# Keyword arguments

$(DOC_INITIAL_WARMUP_ARGS)
"""
function initialize_warmup_state(rng, ℓ; q = random_position(rng, dimension(ℓ)),
                                 κ = GaussianKineticEnergy(dimension(ℓ)), ϵ = nothing)
    WarmupState(evaluate_ℓ(ℓ, q), κ, ϵ)
end

"""
$(TYPEDEF)

Find a local optimum (using quasi-Newton methods).

It is recommended that this stage is applied so that the initial stepsize selection happens
in a region which is at least plausible.
"""
Base.@kwdef struct FindLocalOptimum{T}
    """
    Add `-0.5 * magnitude_penalty * sum(abs2, q)` to the log posterior **when finding the local
    optimum**. This can help avoid getting into high-density edge areas of the posterior
    which are otherwise not typical (eg multilevel models).
    """
    magnitude_penalty::T = 1e-4
    """
    Maximum number of iterations in the optimization algorithm. Recall that we don't need to
    find the mode, or even a local mode, just be in a reasonable region.
    """
    iterations::Int = 50
    # FIXME allow custom algorithm, tolerance, etc
end

function warmup(sampling_logdensity, local_optimization::FindLocalOptimum, warmup_state)
    @unpack ℓ, reporter = sampling_logdensity
    @unpack magnitude_penalty, iterations = local_optimization
    @unpack Q, κ, ϵ = warmup_state
    @unpack q = Q
    report(reporter, "finding initial optimum")
    fg! = function(F, G, q)
        ℓq, ∇ℓq = logdensity_and_gradient(ℓ, q)
        if G ≠ nothing
            @. G = -∇ℓq - q * magnitude_penalty
        end
        -ℓq - (0.5 * magnitude_penalty * sum(abs2, q))
    end
    objective = NLSolversBase.OnceDifferentiable(NLSolversBase.only_fg!(fg!), q)
    opt = Optim.optimize(objective, q, Optim.LBFGS(),
                         Optim.Options(; iterations = iterations))
    q = Optim.minimizer(opt)
    nothing, WarmupState(evaluate_ℓ(ℓ, q), κ, ϵ)
end

function warmup(sampling_logdensity, stepsize_search::InitialStepsizeSearch, warmup_state)
    @unpack rng, ℓ, reporter = sampling_logdensity
    @unpack Q, κ, ϵ = warmup_state
    @argcheck ϵ ≡ nothing "stepsize ϵ manually specified, won't perform initial search"
    z = PhasePoint(Q, rand_p(rng, κ))
    ϵ = find_initial_stepsize(stepsize_search, local_acceptance_ratio(Hamiltonian(κ, ℓ), z))
    report(reporter, "found initial stepsize",
           ϵ = round(ϵ; sigdigits = REPORT_SIGDIGITS))
    nothing, WarmupState(Q, κ, ϵ)
end

"""
$(TYPEDEF)

Tune the step size `ϵ` during sampling, and the metric of the kinetic energy at the end of
the block. The method for the latter is determined by the type parameter `M`, which can be

1. `Diagonal` for diagonal metric (the default),

2. `Symmetric` for a dense metric,

3. `Nothing` for an unchanged metric.

# Results

A `NamedTuple` with the following fields:

- `chain`, a vector of position vectors

- `tree_statistics`, a vector of tree statistics for each sample

- `ϵs`, a vector of step sizes for each sample

# Fields

$(FIELDS)
"""
struct TuningNUTS{M,D}
    "Number of samples."
    N::Int
    "Dual averaging parameters."
    stepsize_adaptation::D
    """
    Regularization factor for normalizing variance. An estimated covariance matrix `Σ` is
    rescaled by `λ` towards ``σ²I``, where ``σ²`` is the median of the diagonal. The
    constructor has a reasonable default.
    """
    λ::Float64
    function TuningNUTS{M}(N::Integer, stepsize_adaptation::D,
                           λ = 5.0/N) where {M <: Union{Nothing,Diagonal,Symmetric},D}
        @argcheck N ≥ 20        # variance estimator is kind of meaningless for few samples
        @argcheck λ ≥ 0
        new{M,D}(N, stepsize_adaptation, λ)
    end
end

function Base.show(io::IO, tuning::TuningNUTS{M}) where {M}
    @unpack N, stepsize_adaptation, λ = tuning
    print(io, "Stepsize and metric tuner, $(N) samples, $(M) metric, regularization $(λ)")
end

"""
$(SIGNATURES)

Form a matrix from positions (`q`), with each column containing a position.
"""
position_matrix(chain) = reduce(hcat, chain)

"""
$(SIGNATURES)

Estimate the inverse metric from the chain.

In most cases, this should be regularized, see [`regularize_M⁻¹`](@ref).
"""
sample_M⁻¹(::Type{Diagonal}, chain) = Diagonal(vec(var(position_matrix(chain); dims = 2)))

sample_M⁻¹(::Type{Symmetric}, chain) = Symmetric(cov(position_matrix(chain); dims = 2))

"""
$(SIGNATURES)

Adjust the inverse metric estimated from the sample, using an *ad-hoc* shrinkage method.
"""
function regularize_M⁻¹(Σ::Union{Diagonal,Symmetric}, λ::Real)
    # ad-hoc “shrinkage estimator”
    (1 - λ) * Σ + λ * UniformScaling(max(1e-3, median(diag(Σ))))
end

function warmup(sampling_logdensity, tuning::TuningNUTS{M}, warmup_state) where {M}
    @unpack rng, ℓ, algorithm, reporter = sampling_logdensity
    @unpack Q, κ, ϵ = warmup_state
    @unpack N, stepsize_adaptation, λ = tuning
    chain = Vector{typeof(Q.q)}(undef, N)
    tree_statistics = Vector{TreeStatisticsNUTS}(undef, N)
    H = Hamiltonian(κ, ℓ)
    ϵ_state = initial_adaptation_state(stepsize_adaptation, ϵ)
    ϵs = Vector{Float64}(undef, N)
    mcmc_reporter = make_mcmc_reporter(reporter, N; tuning = M ≡ Nothing ? "stepsize" :
                                       "stepsize and $(M) metric")
    for i in 1:N
        ϵ = current_ϵ(ϵ_state)
        ϵs[i] = ϵ
        Q, stats = sample_tree(rng, algorithm, H, Q, ϵ)
        chain[i] = Q.q
        tree_statistics[i] = stats
        ϵ_state = adapt_stepsize(stepsize_adaptation, ϵ_state, stats.acceptance_rate)
        report(mcmc_reporter, i; ϵ = round(ϵ; sigdigits = REPORT_SIGDIGITS))
    end
    if M ≢ Nothing
        κ = GaussianKineticEnergy(regularize_M⁻¹(sample_M⁻¹(M, chain), λ))
        report(mcmc_reporter, "adaptation finished", adapted_kinetic_energy = κ)
    end
    ((chain = chain, tree_statistics = tree_statistics, ϵs = ϵs),
     WarmupState(Q, κ, final_ϵ(ϵ_state)))
end

"""
$(TYPEDEF)

A composite type for performing MCMC stepwise after warmup.

The type is *not* part of the API, see [`mcmc_steps`](@ref) and [`mcmc_next_step`](@ref).
"""
struct MCMCSteps{TR,TA,TH,TE}
    rng::TR
    algorithm::TA
    H::TH
    ϵ::TE
end

"""
$(SIGNATURES)

Return a value which can be used to perform MCMC stepwise, eg until some criterion is
satisfied about the sample. See [`mcmc_next_step`](@ref).

Two constructors are available:

1. Explicitly providing
    - `rng`, the random number generator,
    - `algorithm`, see [`mcmc_with_warmup`](@ref),
    - `κ`, the (adapted) metric,
    - `ℓ`, the log density callable (see [`mcmc_with_warmup`](@ref),
    - `ϵ`, the stepsize.

2. Using the fields `sampling_logdensity` and `warmup_state`, eg from
    [`mcmc_keep_warmup`](@ref) (make sure you use eg `final_warmup_state`).

# Example

```julia
# initialization
results = DynamicHMC.mcmc_keep_warmup(RNG, ℓ, 0; reporter = NoProgressReport())
steps = mcmc_steps(results.sampling_logdensity, results.final_warmup_state)
Q = results.final_warmup_state.Q

# a single update step
Q, tree_stats = mcmc_next_step(steps, Q)

# extract the position
Q.q
```
"""
mcmc_steps(rng, algorithm, κ, ℓ, ϵ) = MCMCSteps(rng, algorithm, Hamiltonian(κ, ℓ), ϵ)

function mcmc_steps(sampling_logdensity::SamplingLogDensity, warmup_state)
    @unpack rng, ℓ, algorithm = sampling_logdensity
    @unpack κ, ϵ = warmup_state
    mcmc_steps(rng, algorithm, κ, ℓ, ϵ)
end

"""
$(SIGNATURES)

Given `Q` (an evaluated log density at a position), return the next `Q` and tree statistics.
"""
function mcmc_next_step(mcmc_steps::MCMCSteps, Q::EvaluatedLogDensity)
    @unpack rng, algorithm, H, ϵ = mcmc_steps
    sample_tree(rng, algorithm, H, Q, ϵ)
end

"""
$(SIGNATURES)

Markov Chain Monte Carlo for `sampling_logdensity`, with the adapted `warmup_state`.

Return a `NamedTuple` of

- `chain`, a vector of length `N` that contains the positions,

- `tree_statistics`, a vector of length `N` with the tree statistics.
"""
function mcmc(sampling_logdensity, N, warmup_state)
    @unpack reporter = sampling_logdensity
    @unpack Q = warmup_state
    chain = Vector{typeof(Q.q)}(undef, N)
    tree_statistics = Vector{TreeStatisticsNUTS}(undef, N)
    mcmc_reporter = make_mcmc_reporter(reporter, N)
    steps = mcmc_steps(sampling_logdensity, warmup_state)
    for i in 1:N
        Q, tree_statistics[i] = mcmc_next_step(steps, Q)
        chain[i] = Q.q
        report(mcmc_reporter, i)
    end
    (chain = chain, tree_statistics = tree_statistics)
end

"""
$(SIGNATURES)

Helper function for constructing the “middle” doubling warmup stages in
[`default_warmup_stages`](@ref).
"""
function _doubling_warmup_stages(M, stepsize_adaptation, middle_steps,
                                 doubling_stages::Val{D}) where {D}
    ntuple(i -> TuningNUTS{M}(middle_steps * 2^(i - 1), stepsize_adaptation), D)
end

"""
$(SIGNATURES)

A sequence of warmup stages:

1. find the local optimum using `local_optimization`,

2. select an initial stepsize using `stepsize_search` (default: based on a heuristic),

3. tuning stepsize with `init_steps` steps

4. tuning stepsize and covariance: first with `middle_steps` steps, then repeat with twice
   the steps `doubling_stages` times

5. tuning stepsize with `terminating_steps` steps.

`M` (`Diagonal`, the default or `Symmetric`) determines the type of the metric adapted from
the sample.

This is the suggested tuner of most applications.

Use `nothing` for `local_optimization` or `stepsize_adaptation` to skip the corresponding
step.
"""
function default_warmup_stages(;
                               local_optimization = FindLocalOptimum(),
                               stepsize_search = InitialStepsizeSearch(),
                               M::Type{<:Union{Diagonal,Symmetric}} = Diagonal,
                               stepsize_adaptation = DualAveraging(),
                               init_steps = 75, middle_steps = 25, doubling_stages = 5,
                               terminating_steps = 50)
    (local_optimization, stepsize_search,
     TuningNUTS{Nothing}(init_steps, stepsize_adaptation),
     _doubling_warmup_stages(M, stepsize_adaptation, middle_steps, Val(doubling_stages))...,
     TuningNUTS{Nothing}(terminating_steps, stepsize_adaptation))
end

"""
$(SIGNATURES)

A sequence of warmup stages for fixed stepsize:

1. find the local optimum using `local_optimization`,

2. tuning covariance: first with `middle_steps` steps, then repeat with twice
   the steps `doubling_stages` times

Very similar to [`default_warmup_stages`](@ref), but omits the warmup stages with just
stepsize tuning.
"""
function fixed_stepsize_warmup_stages(;
                                      local_optimization = FindLocalOptimum(),
                                      M::Type{<:Union{Diagonal,Symmetric}} = Diagonal,
                                      middle_steps = 25, doubling_stages = 5)
    (local_optimization,
     _doubling_warmup_stages(M, FixedStepsize(), middle_steps, Val(doubling_stages))...)
end

"""
$(SIGNATURES)

Helper function for implementing warmup.

!!! note
    Changes may imply documentation updates in [`mcmc_keep_warmup`](@ref).
"""
function _warmup(sampling_logdensity, stages, initial_warmup_state)
    foldl(stages; init = ((), initial_warmup_state)) do acc, stage
        stages_and_results, warmup_state = acc
        results, warmup_state′ = warmup(sampling_logdensity, stage, warmup_state)
        stage_information = (stage = stage, results = results, warmup_state = warmup_state′)
        (stages_and_results..., stage_information), warmup_state′
    end
end

"Shared docstring part for the MCMC API."
const DOC_MCMC_ARGS =
"""
# Arguments

- `rng`: the random number generator, eg `Random.GLOBAL_RNG`.

- `ℓ`: the log density, supporting the API of the `LogDensityProblems` package

- `N`: the number of samples for inference, after the warmup.

# Keyword arguments

- `initialization`: see below.

- `warmup_stages`: a sequence of warmup stages. See [`default_warmup_stages`](@ref) and
  [`fixed_stepsize_warmup_stages`](@ref); the latter requires an `ϵ` in initialization.

- `algorithm`: see [`NUTS`](@ref). It is very unlikely you need to modify
  this, except perhaps for the maximum depth.

- `reporter`: how progress is reported. By default, verbosely for interactive sessions using
  the log message mechanism (see [`LogProgressReport`](@ref), and no reporting for
  non-interactive sessions (see [`NoProgressReport`](@ref)).

# Initialization

The `initialization` keyword argument should be a `NamedTuple` which can contain the
following fields (all of them optional and provided with reasonable defaults):

$(DOC_INITIAL_WARMUP_ARGS)
"""

"""
$(SIGNATURES)

Perform MCMC with NUTS, keeping the warmup results. Returns a `NamedTuple` of

- `initial_warmup_state`, which contains the initial warmup state

- `warmup`, an iterable of `NamedTuple`s each containing fields

    - `stage`: the relevant warmup stage

    - `results`: results returned by that warmup stage (may be `nothing` if not applicable,
      or a chain, with tree statistics, etc; see the documentation of stages)

    - `warmup_state`: the warmup state *after* the corresponding stage.

- `final_warmup_state`, which contains the final adaptation after all the warmup

- `inference`, which has `chain` and `tree_statistics`, see [`mcmc_with_warmup`](@ref).

- `sampling_logdensity`, which contains information that is invariant to warmup

!!! warning
    This function is not (yet) exported because the the warmup interface may change with
    minor versions without being considered breaking. Recommended for interactive use.

$(DOC_MCMC_ARGS)
"""
function mcmc_keep_warmup(rng::AbstractRNG, ℓ, N::Integer;
                          initialization = (),
                          warmup_stages = default_warmup_stages(),
                          algorithm = NUTS(),
                          reporter = default_reporter())
    sampling_logdensity = SamplingLogDensity(rng, ℓ, algorithm, reporter)
    initial_warmup_state = initialize_warmup_state(rng, ℓ; initialization...)
    warmup, warmup_state = _warmup(sampling_logdensity, warmup_stages, initial_warmup_state)
    inference = mcmc(sampling_logdensity, N, warmup_state)
    (initial_warmup_state = initial_warmup_state, warmup = warmup,
     final_warmup_state = warmup_state, inference = inference,
     sampling_logdensity = sampling_logdensity)
end

"""
$(SIGNATURES)

Perform MCMC with NUTS, including warmup which is not returned. Return a `NamedTuple` of

- `chain`, a vector of positions from the posterior

- `tree_statistics`, a vector of tree statistics

- `κ` and `ϵ`, the adapted metric and stepsize.

$(DOC_MCMC_ARGS)

# Usage examples

Using a fixed stepsize:
```julia
mcmc_with_warmup(rng, ℓ, N;
                 initialization = (ϵ = 0.1, ),
                 warmup_stages = fixed_stepsize_warmup_stages())
```

Starting from a given position `q₀` and kinetic energy scaled down (will still be adapted):
```julia
mcmc_with_warmup(rng, ℓ, N;
                 initialization = (q = q₀, κ = GaussianKineticEnergy(5, 0.1)))
```

Using a dense metric:
```julia
mcmc_with_warmup(rng, ℓ, N;
                 warmup_stages = default_warmup_stages(; M = Symmetric))
```

Disabling the optimization step:
```julia
mcmc_with_warmup(rng, ℓ, N;
                 warmup_stages = default_warmup_stages(; local_optimization = nothing,
                                                         M = Symmetric))
```
"""
function mcmc_with_warmup(rng, ℓ, N; initialization = (),
                          warmup_stages = default_warmup_stages(),
                          algorithm = NUTS(), reporter = default_reporter())
    @unpack final_warmup_state, inference =
        mcmc_keep_warmup(rng, ℓ, N; initialization = initialization,
                         warmup_stages = warmup_stages, algorithm = algorithm,
                         reporter = reporter)
    @unpack κ, ϵ = final_warmup_state
    (inference..., κ = κ, ϵ = ϵ)
end
