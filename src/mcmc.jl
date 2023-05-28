#####
##### Sampling: high-level interface and building blocks
#####

export InitialStepsizeSearch, DualAveraging, TuningNUTS, mcmc_with_warmup,
    default_warmup_stages, fixed_stepsize_warmup_stages, stack_posterior_matrices,
    pool_posterior_matrices

"Significant digits to display for reporting."
const REPORT_SIGDIGITS = 3

####
#### docstrings for reuse (not exported, internal)
####

const _DOC_POSTERIOR_MATRIX =
    "`posterior_matrix`, a matrix of position vectors, indexes by `[parameter_index, draw_index]`"

const _DOC_TREE_STATISTICS =
    "`tree_statistics`, a vector of tree statistics for each sample"

const _DOC_EPSILONS = "`ϵs`, a vector of step sizes for each sample"

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
    "phasepoint"
    Q::TQ
    "kinetic energy"
    κ::Tκ
    "stepsize"
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
    WarmupState(evaluate_ℓ(ℓ, q; strict = true), κ, ϵ)
end

function warmup(sampling_logdensity, stepsize_search::InitialStepsizeSearch, warmup_state)
    @unpack rng, ℓ, reporter = sampling_logdensity
    @unpack Q, κ, ϵ = warmup_state
    @argcheck ϵ ≡ nothing "stepsize ϵ manually specified, won't perform initial search"
    z = PhasePoint(Q, rand_p(rng, κ))
    try
        ϵ = find_initial_stepsize(stepsize_search, local_log_acceptance_ratio(Hamiltonian(κ, ℓ), z))
        report(reporter, "found initial stepsize",
               ϵ = round(ϵ; sigdigits = REPORT_SIGDIGITS))
        nothing, WarmupState(Q, κ, ϵ)
    catch e
        @info "failed to find initial stepsize" q = z.Q.q p = z.p κ
        rethrow(e)
    end
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

- $(_DOC_POSTERIOR_MATRIX)

- $(_DOC_TREE_STATISTICS)

- $(_DOC_EPSILONS)

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

Estimate the inverse metric from the chain.

In most cases, this should be regularized, see [`regularize_M⁻¹`](@ref).
"""
sample_M⁻¹(::Type{Diagonal}, posterior_matrix) = Diagonal(vec(var(posterior_matrix; dims = 2)))

sample_M⁻¹(::Type{Symmetric}, posterior_matrix) = Symmetric(cov(posterior_matrix; dims = 2))

"""
$(SIGNATURES)

Adjust the inverse metric estimated from the sample, using an *ad-hoc* shrinkage method.
"""
function regularize_M⁻¹(Σ::Union{Diagonal,Symmetric}, λ::Real)
    # ad-hoc “shrinkage estimator”
    (1 - λ) * Σ + λ * UniformScaling(max(1e-3, median(diag(Σ))))
end

"""
$(SIGNATURES)

Create an empty posterior matrix, based on `Q` (a logdensity evaluated at a position).
"""
_empty_posterior_matrix(Q, N) = Matrix{eltype(Q.q)}(undef, length(Q.q), N)

"""
$(SIGNATURES)

Perform a warmup on a given `sampling_logdensity`, using the specified `tuning`, starting
from `warmup_state`.

Return two values. The first is either `nothing`, or a `NamedTuple` of

- $(_DOC_POSTERIOR_MATRIX)

- $(_DOC_TREE_STATISTICS)

- $(_DOC_EPSILONS)

The second is the warmup state.
"""
function warmup(sampling_logdensity, tuning::TuningNUTS{M}, warmup_state) where {M}
    @unpack rng, ℓ, algorithm, reporter = sampling_logdensity
    @unpack Q, κ, ϵ = warmup_state
    @unpack N, stepsize_adaptation, λ = tuning
    posterior_matrix = _empty_posterior_matrix(Q, N)
    tree_statistics = Vector{TreeStatisticsNUTS}(undef, N)
    H = Hamiltonian(κ, ℓ)
    ϵ_state = initial_adaptation_state(stepsize_adaptation, ϵ)
    ϵs = Vector{Float64}(undef, N)
    mcmc_reporter = make_mcmc_reporter(reporter, N;
                                       currently_warmup = true,
                                       tuning = M ≡ Nothing ? "stepsize" : "stepsize and $(M) metric")
    for i in 1:N
        ϵ = current_ϵ(ϵ_state)
        ϵs[i] = ϵ
        Q, stats = sample_tree(rng, algorithm, H, Q, ϵ)
        posterior_matrix[:, i] = Q.q
        tree_statistics[i] = stats
        ϵ_state = adapt_stepsize(stepsize_adaptation, ϵ_state, stats.acceptance_rate)
        report(mcmc_reporter, i; ϵ = round(ϵ; sigdigits = REPORT_SIGDIGITS))
    end
    if M ≢ Nothing
        κ = GaussianKineticEnergy(regularize_M⁻¹(sample_M⁻¹(M, posterior_matrix), λ))
        report(mcmc_reporter, "adaptation finished", adapted_kinetic_energy = κ)
    end
    ((; posterior_matrix, tree_statistics, ϵs), WarmupState(Q, κ, final_ϵ(ϵ_state)))
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

- $(_DOC_POSTERIOR_MATRIX)

- $(_DOC_TREE_STATISTICS)
"""
function mcmc(sampling_logdensity, N, warmup_state)
    @unpack reporter = sampling_logdensity
    @unpack Q = warmup_state
    posterior_matrix = _empty_posterior_matrix(Q, N)
    tree_statistics = Vector{TreeStatisticsNUTS}(undef, N)
    mcmc_reporter = make_mcmc_reporter(reporter, N; currently_warmup = false)
    steps = mcmc_steps(sampling_logdensity, warmup_state)
    for i in 1:N
        Q, tree_statistics[i] = mcmc_next_step(steps, Q)
        posterior_matrix[:, i] = Q.q
        report(mcmc_reporter, i)
    end
    (; posterior_matrix, tree_statistics)
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

1. select an initial stepsize using `stepsize_search` (default: based on a heuristic),

2. tuning stepsize with `init_steps` steps

3. tuning stepsize and covariance: first with `middle_steps` steps, then repeat with twice
   the steps `doubling_stages` times

4. tuning stepsize with `terminating_steps` steps.

`M` (`Diagonal`, the default or `Symmetric`) determines the type of the metric adapted from
the sample.

This is the suggested tuner of most applications.

Use `nothing` for `stepsize_adaptation` to skip the corresponding step.
"""
function default_warmup_stages(;
                               stepsize_search = InitialStepsizeSearch(),
                               M::Type{<:Union{Diagonal,Symmetric}} = Diagonal,
                               stepsize_adaptation = DualAveraging(),
                               init_steps = 75, middle_steps = 25, doubling_stages = 5,
                               terminating_steps = 50)
    (stepsize_search,
     TuningNUTS{Nothing}(init_steps, stepsize_adaptation),
     _doubling_warmup_stages(M, stepsize_adaptation, middle_steps, Val(doubling_stages))...,
     TuningNUTS{Nothing}(terminating_steps, stepsize_adaptation))
end

"""
$(SIGNATURES)

A sequence of warmup stages for fixed stepsize, only tuning covariance: first with
`middle_steps` steps, then repeat with twice the steps `doubling_stages` times

Very similar to [`default_warmup_stages`](@ref), but omits the warmup stages with just
stepsize tuning.
"""
function fixed_stepsize_warmup_stages(;
                                      M::Type{<:Union{Diagonal,Symmetric}} = Diagonal,
                                      middle_steps = 25, doubling_stages = 5)
    _doubling_warmup_stages(M, FixedStepsize(), middle_steps, Val(doubling_stages))
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
        stage_information = (stage, results, warmup_state = warmup_state′)
        (stages_and_results..., stage_information), warmup_state′
    end
end

"Shared docstring part for the MCMC API."
const DOC_MCMC_ARGS =
"""
# Arguments

- `rng`: the random number generator, eg `Random.default_rng()`.

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

- `inference`, which has `posterior_matrix` and `tree_statistics`, see
  [`mcmc_with_warmup`](@ref).

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
    (; initial_warmup_state, warmup, final_warmup_state = warmup_state, inference,
     sampling_logdensity)
end

"""
$(SIGNATURES)

Perform MCMC with NUTS, including warmup which is not returned. Return a `NamedTuple` of

- $(_DOC_POSTERIOR_MATRIX)

- $(_DOC_TREE_STATISTICS)

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

Disabling the initial stepsize search (provided explicitly, still adapted):
```julia
mcmc_with_warmup(rng, ℓ, N;
                 initialization = (ϵ = 1.0, ),
                 warmup_stages = default_warmup_stages(; stepsize_search = nothing))
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
    (; inference..., κ, ϵ)
end

####
#### utilities
####

"""
$(SIGNATURES)

Given a vector of `results`, each containing a property `posterior_matrix` (eg obtained from
[`mcmc_with_warmup`](@ref) with the same sample length), return a lazy view as an array
indexed by `[draw_index, chain_index, parameter_index]`.

This is useful as an input for eg `MCMCDiagnosticTools.ess_rhat`.

!!! note
    The ordering is not compatible with MCMCDiagnostictools version < 0.2.
"""
function stack_posterior_matrices(results)
    @cast _[i, k, j]:= results[k].posterior_matrix[j, i]
end

"""
$(SIGNATURES)

Given a vector of `results`, each containing a property `posterior_matrix` (eg obtained from
[`mcmc_with_warmup`](@ref) with the same sample length), return a lazy view as an array
indexed by `[parameter_index, pooled_draw_index]`.

This is useful for posterior analysis after diagnostics (see eg `Base.eachcol`).
"""
function pool_posterior_matrices(results)
    @cast _[i, j ⊗ k] := results[k].posterior_matrix[i, j]
end
