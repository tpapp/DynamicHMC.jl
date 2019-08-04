#####
##### Sampling: high-level interface and building blocks
#####

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
struct SamplingLogDensity{R,L,O}
    "Random number generator."
    rng::R
    "Log density."
    ℓ::L
    "Sampler options."
    sampler_options::O
end

####
#### warmup building blocks
####

###
### warmup state
###

struct WarmupState{TQ <: EvaluatedLogDensity,Tκ <: KineticEnergy, Tϵ <: Union{Real,Nothing}}
    Q::TQ
    κ::Tκ
    ϵ::Tϵ
end

function Base.show(io::IO, warmup_state::WarmupState)
    @unpack κ, ϵ = warmup_state
    print(io, "adapted sampling parameters: stepsize (ϵ) ≈ $(round(ϵ; sigdigits = 3))\n",
          "  $(κ)")
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

- `ϵ`: initial stepsize, or `nothing` for heuristic finders.
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
    @unpack ℓ = sampling_logdensity
    @unpack Q, κ, ϵ = warmup_state
    @unpack q = Q
    fg! = function(F, G, q)
        ℓq, ∇ℓq = logdensity_and_gradient(ℓ, q)
        if G ≠ nothing
            @. G = -∇ℓq
        end
        -ℓq
    end
    objective = NLSolversBase.OnceDifferentiable(NLSolversBase.only_fg!(fg!), q)
    opt = Optim.optimize(objective, q, Optim.LBFGS())
    Optim.converged(opt) || error("could not find local optimum of log density")
    q = Optim.minimizer(opt)
    nothing, WarmupState(evaluate_ℓ(ℓ, q), κ, ϵ)
end

function warmup(sampling_logdensity, stepsize_search::InitialStepsizeSearch, warmup_state)
    @unpack rng, ℓ = sampling_logdensity
    @unpack Q, κ, ϵ = warmup_state
    @argcheck ϵ ≡ nothing "stepsize ϵ manually specified, won't perform initial search"
    z = PhasePoint(Q, rand_p(rng, κ))
    ϵ = find_initial_stepsize(stepsize_search, local_acceptance_ratio(Hamiltonian(κ, ℓ), z))
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
struct TuningNUTS{M,D <: DualAveragingParameters}
    "Number of samples."
    N::Int
    "Dual averaging parameters."
    dual_averaging::D
    """
    Regularization factor for normalizing variance. An estimated covariance matrix `Σ` is
    rescaled by `λ`` towards `σ²I`, where `σ²` is the median of the diagonal. The
    constructor has a reasonable default.
    """
    λ::Float64
    function TuningNUTS{M}(N::Integer, dual_averaging::D,
                           λ = 5.0/N) where {M <: Union{Nothing,Diagonal,Symmetric},D}
        @argcheck N ≥ 20        # variance estimator is kind of meaningless for few samples
        @argcheck λ ≥ 0
        new{M,D}(N, dual_averaging, λ)
    end
end

function Base.show(io::IO, tuning::TuningNUTS{M}) where {M}
    @unpack N, dual_averaging, λ = tuning
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
    @unpack rng, ℓ, sampler_options = sampling_logdensity
    @unpack Q, κ, ϵ = warmup_state
    @unpack N, dual_averaging, λ = tuning
    chain = Vector{typeof(Q.q)}(undef, N)
    tree_statistics = Vector{TreeStatisticsNUTS}(undef, N)
    H = Hamiltonian(κ, ℓ)
    logϵ_adaptation = DualAveragingAdaptation(log(ϵ))
    ϵs = Vector{Float64}(undef, N)
    for i in 1:N
        ϵ = current_ϵ(logϵ_adaptation)
        ϵs[i] = ϵ
        Q, stats = NUTS_sample_tree(rng, sampler_options, H, Q, ϵ)
        chain[i] = Q.q
        tree_statistics[i] = stats
        logϵ_adaptation = adapt_stepsize(dual_averaging, logϵ_adaptation,
                                         stats.acceptance_statistic)
    end
    if M ≢ Nothing
        κ = GaussianKineticEnergy(regularize_M⁻¹(sample_M⁻¹(M, chain), λ))
    end
    ((chain = chain, tree_statistics = tree_statistics, ϵs = ϵs),
     WarmupState(Q, κ, final_ϵ(logϵ_adaptation)))
end

"""
$(SIGNATURES)

Markov Chain Monte Carlo for `sampling_logdensity`, with the adapted `warmup_state`.

Return a `NamedTuple` of

- `chain`, a vector of length `N` that contains the positions,

- `tree_statistics`, a vector of length `N` with the tree statistics.
"""
function mcmc(sampling_logdensity, N, warmup_state)
    @unpack rng, ℓ, sampler_options = sampling_logdensity
    @unpack Q, κ, ϵ = warmup_state
    chain = Vector{typeof(Q.q)}(undef, N)
    tree_statistics = Vector{TreeStatisticsNUTS}(undef, N)
    H = Hamiltonian(κ, ℓ)
    for i in 1:N
        Q, tree_statistics[i] = NUTS_sample_tree(rng, sampler_options, H, Q, ϵ)
        chain[i] = Q.q
    end
    (chain = chain, tree_statistics = tree_statistics)
end

"""
$(SIGNATURES)

Helper function for constructing the “middle” doubling warmup stages in
[`default_warmup_stages`](@ref).
"""
function _doubling_warmup_stages(M, dual_averaging, middle_steps,
                                 doubling_stages::Val{D}) where {D}
    ntuple(i -> TuningNUTS{M}(middle_steps * 2^(i - 1), dual_averaging), D)
end

"""
$(SIGNATURES)

A sequence of warmup stages:

1. tuning stepsize with `init_steps` steps

2. tuning stepsize and covariance: first with `middle_steps` steps, then repeat with twice
   the steps `doubling_stages` times

3. tuning stepsize with `terminating_steps` steps.

`M` (`Diagonal`, the default or `Symmetric`) determines the type of the metric adapted from
the sample.

(This is the suggested tuner of most papers on NUTS).
"""
function default_warmup_stages(;
                               M::Type{<:Union{Diagonal,Symmetric}} = Diagonal,
                               dual_averaging = DualAveragingParameters(),
                               init_steps = 75, middle_steps = 25, doubling_stages = 5,
                               terminating_steps = 50)
    (FindLocalOptimum(),
     InitialStepsizeSearch(),
     TuningNUTS{Nothing}(init_steps, dual_averaging),
     _doubling_warmup_stages(M, dual_averaging, middle_steps, Val(doubling_stages))...,
     TuningNUTS{Nothing}(terminating_steps, dual_averaging))
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
- `rng`: the random number generator, eg `Random.GLOBAL_RNG`

- `ℓ`: the log density, supporting the API of the `LogDensityProblems` package

- `N`: the number of samples for inference, after the warmup.
"""

"""
$(SIGNATURES)

Perform MCMC with NUTS, keeping the warmup results. Returns a `NamedTuple` of

- `warmup`, which contains all the warmup information and diagnostics

- `warmup_state`, which contains the final adaptation after all the warmup

- `inference`, which has `chain` and `tree_statistics`, see [`mcmc_with_warmup`](@ref).

!!! warning
    This function is not (yet) exported because the the warmup API may change.

# Arguments

$(DOC_MCMC_ARGS)

# Keyword arguments

$(DOC_INITIAL_WARMUP_ARGS)
"""
function mcmc_keep_warmup(rng::AbstractRNG, ℓ, N::Integer;
                          initialization = (),
                          warmup_stages = default_warmup_stages(),
                          sampler_options = TreeOptionsNUTS())
    sampling_logdensity = SamplingLogDensity(rng, ℓ, sampler_options)
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

# Arguments

$(DOC_MCMC_ARGS)

# Keyword arguments

$(DOC_INITIAL_WARMUP_ARGS)
"""
function mcmc_with_warmup(rng, ℓ, N; initialization = (),
                          warmup_stages = default_warmup_stages(),
                          sampler_options = TreeOptionsNUTS())
    @unpack warmup_state, inference =
        mcmc_keep_warmup(rng, ℓ, N; initialization = initialization,
                         warmup_stages = warmup_stages, sampler_options = sampler_options)
    @unpack κ, ϵ = warmup_state
    (inference..., κ = κ, ϵ = ϵ)
end
