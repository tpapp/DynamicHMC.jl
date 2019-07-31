#####
##### Sampling: high-level interface and building blocks
#####

export AdaptationState, mcmc_stage, FindLocalOptimum, TuningNUTS, SamplingNUTS,
    mcmc_NUTS_adaptation, mcmc_NUTS


####
#### adaptation building blocks
####

###
### adaptation state
###

struct AdaptationState{TQ <: EvaluatedLogDensity,Tκ <: KineticEnergy, Tϵ <: Union{Real,Nothing}}
    Q::TQ
    κ::Tκ
    ϵ::Tϵ
end

function Base.show(io::IO, adaptation_state::AdaptationState)
    @unpack κ, ϵ = adaptation_state
    print(io, "adapted sampling parameters: stepsize (ϵ) ≈ $(round(ϵ; sigdigits = 3))\n",
          "  $(κ)")
end

###
### adaptation interface and stages
###

"""
```julia
samples, statistics, adaptation_state′ =
    mcmc_stage(rng, ℓ, sampler_options, adaptation_state, stage)
```
"""
function mcmc_stage end

"""
$(SIGNATURES)

Empty vectors for the first two values returned by [`mcmc_stage`](@ref).
"""
_empty_samples_and_statistics() = Vector{Vector}(), Vector{TreeStatisticsNUTS}()

"""
$(SIGNATURES)

Helper function to create random starting positions in the `[-2,2]ⁿ` box.
"""
random_position(rng, N) = rand(rng, N) .* 4 .- 2

const INITIAL_ADAPTATION_ARGS =
"""
- `q`: initial position. *Default*: random (uniform [-2,2] for each coordinate).

- `κ`: kinetic energy specification. *Default*: Gaussian with identity matrix.

- `ϵ`: initial stepsize, or `nothing` for heuristic finders.
"""

"""
$(SIGNATURES)

Create an initial adaptation state from a random position.

# Keyword arguments

$(INITIAL_ADAPTATION_ARGS)
"""
function initial_adaptation_state(rng, ℓ;
                                  q = random_position(rng, dimension(ℓ)),
                                  κ = GaussianKineticEnergy(dimension(ℓ)),
                                  ϵ = nothing)
    AdaptationState(evaluate_ℓ(ℓ, q), κ, ϵ)
end

"""
$(TYPEDEF)

The a local optimum (using quasi-Newton methods).
"""
struct FindLocalOptimum end     # FIXME allow custom algorithm, tolerance, etc

function mcmc_stage(_, ℓ, _, adaptation_state, local_optimization::FindLocalOptimum)
    @unpack Q, κ, ϵ = adaptation_state
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
    _empty_samples_and_statistics()..., AdaptationState(evaluate_ℓ(ℓ, q), κ, ϵ)
end

function mcmc_stage(rng, ℓ, _, adaptation_state, stepsize_search::InitialStepsizeSearch)
    @unpack Q, κ, ϵ = adaptation_state
    @argcheck ϵ ≡ nothing "stepsize ϵ manually specified, won't perform initial search"
    z = PhasePoint(Q, rand_p(rng, κ))
    ϵ = find_initial_stepsize(stepsize_search, local_acceptance_ratio(Hamiltonian(κ, ℓ), z))
    _empty_samples_and_statistics()..., AdaptationState(Q, κ, ϵ)
end

"""
$(TYPEDEF)


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
position_matrix(sample) = reduce(hcat, sample)

sample_M⁻¹(::Type{Diagonal}, sample) = Diagonal(vec(var(position_matrix(sample); dims = 2)))

sample_M⁻¹(::Type{Symmetric}, sample) = Symmetric(cov(position_matrix(sample); dims = 2))

function regularize_matrix(Σ::Union{Diagonal,Symmetric}, λ::Real)
    # ad-hoc “shrinkage estimator”
    (1 - λ) * Σ + λ * UniformScaling(max(1e-3, median(diag(Σ))))
end

function mcmc_stage(rng, ℓ, sampler_options, adaptation_state, tuning::TuningNUTS{M}) where {M}
    @unpack Q, κ, ϵ = adaptation_state
    @unpack N, dual_averaging, λ = tuning
    sample = Vector{typeof(Q.q)}(undef, N)
    statistics = Vector{TreeStatisticsNUTS}(undef, N)
    H = Hamiltonian(κ, ℓ)
    logϵ_adaptation = DualAveragingAdaptation(log(ϵ))
    for i in 1:N
        Q, stats = NUTS_sample_tree(rng, sampler_options, H, Q, current_ϵ(logϵ_adaptation))
        sample[i] = Q.q
        statistics[i] = stats
        logϵ_adaptation = adapt_stepsize(dual_averaging, logϵ_adaptation,
                                         stats.acceptance_statistic)
    end
    if M ≢ Nothing
        κ = GaussianKineticEnergy(regularize_matrix(sample_M⁻¹(M, sample), λ))
    end
    sample, statistics, AdaptationState(Q, κ, final_ϵ(logϵ_adaptation))
end

struct SamplingNUTS
    N::Int
end

function mcmc_stage(rng, ℓ, sampler_options, adaptation_state, sampling::SamplingNUTS)
    @unpack Q, κ, ϵ = adaptation_state
    @unpack N = sampling
    sample = Vector{typeof(Q.q)}(undef, N)
    statistics = Vector{TreeStatisticsNUTS}(undef, N)
    H = Hamiltonian(κ, ℓ)
    for i in 1:N
        Q, statistics[i] = NUTS_sample_tree(rng, sampler_options, H, Q, ϵ)
        sample[i] = Q.q
    end
    sample, statistics, AdaptationState(Q, κ, ϵ)
end

"""
$(SIGNATURES)

A sequence of adaptation stages:

1. tuning stepsize with `init_steps` steps

2. tuning stepsize and covariance: first with `middle_steps` steps, then repeat with twice
   the steps `doubling_stages` times

3. tuning stepsize with `terminating_steps` steps.

`M` (`Diagonal`, the default or `Symmetric`) determines the type of the metric adapted from
the sample.

(This is the suggested tuner of most papers on NUTS).
"""
function default_adapting_stages(;
                                 M::Type{<:Union{Diagonal,Symmetric}} = Diagonal,
                                 init_steps = 75, middle_steps = 25, doubling_stages = 5,
                                 terminating_steps = 50,
                                 dual_averaging = DualAveragingParameters())
    middle_stages = Any[]
    for _ in 1:doubling_stages
        stages = push!(middle_stages, TuningNUTS{M}(middle_steps, dual_averaging))
        middle_steps *= 2
    end
    vcat(FindLocalOptimum(),
         InitialStepsizeSearch(),
         TuningNUTS{Nothing}(init_steps, dual_averaging),
         middle_stages,
         TuningNUTS{Nothing}(terminating_steps, dual_averaging))
end

function mcmc_stages(rng, ℓ, sampler_options, adaptation_state, stages)
    [begin
     samples, statistics, adaptation_state = mcmc_stage(rng, ℓ, sampler_options,
                                                        adaptation_state, stage)
     (samples = samples, statistics = statistics, adaptation_state = adaptation_state)
     end for stage in stages], adaptation_state
end

function mcmc_NUTS_adaptation(rng, ℓ, N;
                              initialization = (),
                              adapting_stages = default_adapting_stages(),
                              sampler_options = TreeOptionsNUTS())
    adaptation_state = initial_adaptation_state(rng, ℓ, initialization...)
    adaptation, adaptation_state =
        mcmc_stages(rng, ℓ, sampler_options, adaptation_state, adapting_stages)
    inference_samples, inference_statistics, _ =
        mcmc_stage(rng, ℓ, sampler_options, adaptation_state, SamplingNUTS(N))
    (adaptation = adaptation,
     inference = (samples = inference_samples, statistics = inference_statistics))
end

function mcmc_NUTS(rng, ℓ, N;
                   initialization = (),
                   adapting_stages = default_adapting_stages(),
                   sampler_options = TreeOptionsNUTS())
    mcmc_NUTS_adaptation(rng, ℓ, N; initialization = initialization,
                         adapting_stages = adapting_stages,
                         sampler_options = sampler_options).inference
end
