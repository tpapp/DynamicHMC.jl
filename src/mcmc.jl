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
    "Sampler options."
    sampler_options::O
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
    $(FUNCTIONNAME)(sampling_logdensity::SamplingLogDensity, warmup_stage, warmup_state)

Return the *results* and the *next warmup state* after warming up/adapting according to
`warmup_stage`, starting from `warmup_state`.
"""
function warmup end

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
function initial_warmup_state(rng, ℓ; q = random_position(rng, dimension(ℓ)),
                              κ = GaussianKineticEnergy(dimension(ℓ)), ϵ = nothing)
    WarmupState(evaluate_ℓ(ℓ, q), κ, ϵ)
end

"""
$(TYPEDEF)

Find a local optimum (using quasi-Newton methods).

It is recommended that this stage is applied so that the initial stepsize selection happens
in a region which is at least plausible.
"""
struct FindLocalOptimum end     # FIXME allow custom algorithm, tolerance, etc

function warmup(sampling_logdensity, local_optimization::FindLocalOptimum, warmup_state)
    @unpack ℓ, reporter = sampling_logdensity
    @unpack Q, κ, ϵ = warmup_state
    @unpack q = Q
    report(reporter, "finding initial optimum")
    fg! = function(F, G, q)
        ℓq, ∇ℓq = logdensity_and_gradient(ℓ, q)
        if G ≠ nothing
            @. G = -∇ℓq
        end
        -ℓq
    end
    objective = NLSolversBase.OnceDifferentiable(NLSolversBase.only_fg!(fg!), q)
    opt = Optim.optimize(objective, q, Optim.LBFGS())
    Optim.converged(opt) || @warn("could not find local optimum of log density")
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
    @unpack rng, ℓ, sampler_options, reporter = sampling_logdensity
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
        Q, stats = NUTS_sample_tree(rng, sampler_options, H, Q, ϵ)
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
$(SIGNATURES)

Markov Chain Monte Carlo for `sampling_logdensity`, with the adapted `warmup_state`.

Return a `NamedTuple` of

- `chain`, a vector of length `N` that contains the positions,

- `tree_statistics`, a vector of length `N` with the tree statistics.
"""
function mcmc(sampling_logdensity, N, warmup_state)
    @unpack rng, ℓ, sampler_options, reporter = sampling_logdensity
    @unpack Q, κ, ϵ = warmup_state
    chain = Vector{typeof(Q.q)}(undef, N)
    tree_statistics = Vector{TreeStatisticsNUTS}(undef, N)
    H = Hamiltonian(κ, ℓ)
    mcmc_reporter = make_mcmc_reporter(reporter, N)
    for i in 1:N
        Q, tree_statistics[i] = NUTS_sample_tree(rng, sampler_options, H, Q, ϵ)
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

1. find the local optimum,

2. select an initial stepsize based on a heuristic,

3. tuning stepsize with `init_steps` steps

4. tuning stepsize and covariance: first with `middle_steps` steps, then repeat with twice
   the steps `doubling_stages` times

5. tuning stepsize with `terminating_steps` steps.

`M` (`Diagonal`, the default or `Symmetric`) determines the type of the metric adapted from
the sample.

This is the suggested tuner of most applications.
"""
function default_warmup_stages(;
                               M::Type{<:Union{Diagonal,Symmetric}} = Diagonal,
                               stepsize_adaptation = DualAveraging(),
                               init_steps = 75, middle_steps = 25, doubling_stages = 5,
                               terminating_steps = 50)
    (FindLocalOptimum(),
     InitialStepsizeSearch(),
     TuningNUTS{Nothing}(init_steps, stepsize_adaptation),
     _doubling_warmup_stages(M, stepsize_adaptation, middle_steps, Val(doubling_stages))...,
     TuningNUTS{Nothing}(terminating_steps, stepsize_adaptation))
end

"""
$(SIGNATURES)

A sequence of warmup stages for fixed stepsize:

1. find the local optimum,

2. tuning covariance: first with `middle_steps` steps, then repeat with twice
   the steps `doubling_stages` times

Very similar to [`default_warmup_stages`](@ref), but omits the warmup stages with just
stepsize tuning.
"""
function fixed_stepsize_warmup_stages(;
                                      M::Type{<:Union{Diagonal,Symmetric}} = Diagonal,
                                      middle_steps = 25, doubling_stages = 5)
    (FindLocalOptimum(),
     _doubling_warmup_stages(M, FixedStepsize(), middle_steps, Val(doubling_stages))...)
end

function _warmup(sampling_logdensity, stages, initial_state)
    foldl(stages; init = ((), initial_state)) do acc, stage
        stages_and_results, warmup_state = acc
        results, warmup_state′ = warmup(sampling_logdensity, stage, warmup_state)
        stage_information = (stage = stage, results = results, warmup_state = warmup_state)
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

- `sampler_options`: see [`TreeOptionsNUTS`](@ref). It is very unlikely you need to modify
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

- `warmup`, which contains all the warmup information and diagnostics

- `warmup_state`, which contains the final adaptation after all the warmup

- `inference`, which has `chain` and `tree_statistics`, see [`mcmc_with_warmup`](@ref).

!!! warning
    This function is not (yet) exported because the the warmup API may change.

$(DOC_MCMC_ARGS)
"""
function mcmc_keep_warmup(rng::AbstractRNG, ℓ, N::Integer;
                          initialization = (),
                          warmup_stages = default_warmup_stages(),
                          sampler_options = TreeOptionsNUTS(),
                          reporter = default_reporter())
    sampling_logdensity = SamplingLogDensity(rng, ℓ, sampler_options, reporter)
    initial_state = initial_warmup_state(rng, ℓ; initialization...)
    warmup, warmup_state = _warmup(sampling_logdensity, warmup_stages, initial_state)
    inference = mcmc(sampling_logdensity, N, warmup_state)
    (warmup = warmup, warmup_state = warmup_state, inference = inference)
end

"""
$(SIGNATURES)

Perform MCMC with NUTS, including warmup which is not returned. Return a `NamedTuple` of

- `chain`, a vector of positions from the posterior

- `tree_statistics`, a vector of tree statistics

- `κ` and `ϵ`, the adapted metric and stepsize.

$(DOC_MCMC_ARGS)
"""
function mcmc_with_warmup(rng, ℓ, N; initialization = (),
                          warmup_stages = default_warmup_stages(),
                          sampler_options = TreeOptionsNUTS(),
                          reporter = default_reporter())
    @unpack warmup_state, inference =
        mcmc_keep_warmup(rng, ℓ, N; initialization = initialization,
                         warmup_stages = warmup_stages, sampler_options = sampler_options,
                         reporter = reporter)
    @unpack κ, ϵ = warmup_state
    (inference..., κ = κ, ϵ = ϵ)
end
