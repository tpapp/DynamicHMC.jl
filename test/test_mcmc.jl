using DynamicHMC: mcmc_steps, mcmc_next_step, mcmc_keep_warmup, WarmupState

#####
##### Test building blocks of MCMC
#####

@testset "printing" begin
    ℓ = multivariate_normal(ones(1))
    κ = GaussianKineticEnergy(1)
    Q = evaluate_ℓ(ℓ, [1.0])
    @test repr(WarmupState(Q, κ, 1.0)) isa String
    @test repr(WarmupState(Q, κ, nothing)) isa String
end

@testset "mcmc" begin
    ℓ = multivariate_normal(ones(5))

    @testset "default warmup" begin
        results = mcmc_with_warmup(RNG, ℓ, 10000; reporter = NoProgressReport())
        Z = results.posterior_matrix
        @test map(z -> LogDensityProblems.logdensity(ℓ, z), eachcol(Z)) ≈ results.logdensities
        @test norm(mean(Z; dims = 2) .- ones(5), Inf) < 0.04
        @test norm(std(Z; dims = 2) .- ones(5), Inf) < 0.04
        @test mean(x -> x.acceptance_rate, results.tree_statistics) ≥ 0.8
        @test 0.5 ≤ results.ϵ ≤ 2
    end

    @testset "fixed stepsize" begin
        results = mcmc_with_warmup(RNG, ℓ, 10000;
                                   initialization = (ϵ = 1.0, ),
                                   reporter = NoProgressReport(),
                                   warmup_stages = fixed_stepsize_warmup_stages())
        Z = results.posterior_matrix
        @test norm(mean(Z; dims = 2) .- ones(5), Inf) < 0.04
        @test norm(std(Z; dims = 2) .- ones(5), Inf) < 0.04
        @test mean(x -> x.acceptance_rate, results.tree_statistics) ≥ 0.7
    end

    @testset "explicitly provided initial stepsize" begin
        results = mcmc_with_warmup(RNG, ℓ, 10000;
                                   initialization = (ϵ = 1.0, ),
                                   reporter = NoProgressReport(),
                                   warmup_stages = default_warmup_stages(; stepsize_search = nothing))
        Z = results.posterior_matrix
        @test norm(mean(Z; dims = 2) .- ones(5), Inf) < 0.03
        @test norm(std(Z; dims = 2) .- ones(5), Inf) < 0.04
        @test mean(x -> x.acceptance_rate, results.tree_statistics) ≥ 0.7
    end

    @testset "stepwise" begin
        results = mcmc_keep_warmup(RNG, ℓ, 0; reporter = NoProgressReport())
        steps = mcmc_steps(results.sampling_logdensity, results.final_warmup_state)
        qs = let Q = results.final_warmup_state.Q
            [(Q = first(mcmc_next_step(steps, Q)); Q.q) for _ in 1:1000]
        end
        @test norm(mean(reduce(hcat, qs); dims = 2) .- ones(5), Inf) ≤ 0.1
    end
end

@testset "robust U-turn tests" begin
    # Cf https://github.com/tpapp/DynamicHMC.jl/issues/115
    function count_max_depth(rng, ℓ, max_depth; N = 1000)
        results = mcmc_with_warmup(rng, ℓ, N;
                                   algorithm = DynamicHMC.NUTS(max_depth = max_depth),
                                   reporter = NoProgressReport())
        sum(getfield.(results.tree_statistics, :depth) .≥ max_depth)
    end
    ℓ = multivariate_normal(zeros(200))
    max_depth = 12
    M = sum([count_max_depth(RNG, ℓ, max_depth) for _ in 1:20])
    @test M == 0
end

@testset "Float32 support" begin
    # Float32 multivariate normal: ℓ(q) = -½ (q - μ)ᵀ Σ⁻¹ (q - μ) with Σ = I
    struct Float32Normal{V <: AbstractVector}
        μ::V
    end
    LogDensityProblems.capabilities(::Type{<:Float32Normal}) = LogDensityProblems.LogDensityOrder{1}()
    LogDensityProblems.dimension(ℓ::Float32Normal) = length(ℓ.μ)
    function LogDensityProblems.logdensity_and_gradient(ℓ::Float32Normal, q::AbstractVector)
        r = q - ℓ.μ
        T = eltype(q)
        T(-dot(r, r) / 2), -r
    end

    @testset "type propagation" begin
        ℓ32 = Float32Normal(zeros(Float32, 3))
        q0 = randn(Float32, 3)
        results = mcmc_with_warmup(RNG, ℓ32, 100;
                                   initialization = (q = q0,),
                                   reporter = NoProgressReport())
        @test eltype(results.posterior_matrix) == Float32
        @test eltype(results.logdensities) == Float32
        @test results.tree_statistics[1].π isa Float32
        @test results.tree_statistics[1].acceptance_rate isa Float32
        @test results.ϵ isa Float32
    end

    @testset "no type promotion in compute" begin
        # A log density that errors if position is not Float32,
        # catching any accidental promotion in leapfrog/adaptation
        struct StrictFloat32Normal{V <: AbstractVector{Float32}}
            μ::V
        end
        LogDensityProblems.capabilities(::Type{<:StrictFloat32Normal}) = LogDensityProblems.LogDensityOrder{1}()
        LogDensityProblems.dimension(ℓ::StrictFloat32Normal) = length(ℓ.μ)
        function LogDensityProblems.logdensity_and_gradient(ℓ::StrictFloat32Normal, q::AbstractVector)
            @assert eltype(q) === Float32 "position promoted to $(eltype(q)), expected Float32"
            r = q - ℓ.μ
            Float32(-dot(r, r) / 2), -r
        end
        ℓ_strict = StrictFloat32Normal(zeros(Float32, 3))
        q0 = randn(Float32, 3)
        # runs full warmup (stepsize search + dual averaging + metric adaptation)
        # and inference — any Float64 promotion in leapfrog would trigger the assertion
        results = mcmc_with_warmup(RNG, ℓ_strict, 100;
                                   initialization = (q = q0,),
                                   reporter = NoProgressReport())
        @test eltype(results.posterior_matrix) == Float32
        @test results.ϵ isa Float32
    end

    @testset "sample correctness" begin
        μ = Float32[1.0, -0.5, 2.0, 0.0, -1.5]
        ℓ32 = Float32Normal(μ)
        q0 = randn(Float32, 5)
        results = mcmc_with_warmup(RNG, ℓ32, 10000;
                                   initialization = (q = q0,),
                                   reporter = NoProgressReport())
        Z = results.posterior_matrix
        @test eltype(Z) == Float32
        @test norm(mean(Z; dims = 2) .- μ, Inf) < 0.06
        @test norm(std(Z; dims = 2) .- ones(Float32, 5), Inf) < 0.06
        @test mean(x -> x.acceptance_rate, results.tree_statistics) ≥ 0.7
    end

    @testset "fixed stepsize" begin
        ℓ32 = Float32Normal(ones(Float32, 3))
        q0 = randn(Float32, 3)
        results = mcmc_with_warmup(RNG, ℓ32, 5000;
                                   initialization = (q = q0, ϵ = Float32(1.0)),
                                   warmup_stages = fixed_stepsize_warmup_stages(),
                                   reporter = NoProgressReport())
        Z = results.posterior_matrix
        @test eltype(Z) == Float32
        @test norm(mean(Z; dims = 2) .- ones(Float32, 3), Inf) < 0.1
    end

    @testset "stepwise" begin
        ℓ32 = Float32Normal(zeros(Float32, 3))
        q0 = randn(Float32, 3)
        results = mcmc_keep_warmup(RNG, ℓ32, 0;
                                   initialization = (q = q0,),
                                   reporter = NoProgressReport())
        steps = mcmc_steps(results.sampling_logdensity, results.final_warmup_state)
        Q = results.final_warmup_state.Q
        @test eltype(Q.q) == Float32
        qs = [(Q = first(mcmc_next_step(steps, Q)); Q.q) for _ in 1:1000]
        @test eltype(qs[1]) == Float32
        @test norm(mean(reduce(hcat, qs); dims = 2), Inf) ≤ 0.15
    end
end

@testset "posterior accessors sanity checks" begin
    D, N, K = 5, 100, 7
    ℓ = multivariate_normal(ones(5))
    results = fill(mcmc_with_warmup(RNG, ℓ, N; reporter = NoProgressReport()), K)
    @test size(stack_posterior_matrices(results)) == (N, K, D)
    @test size(pool_posterior_matrices(results)) == (D, N * K)
end

# @testset "tuner framework" begin
#     s = StepsizeTuner(10)
#     @test length(s) == 10
#     @test repr(s) == "Stepsize tuner, 10 samples"
#     c = StepsizeCovTuner(19, 7.0)
#     @test length(c) == 19
#     @test repr(c) ==
#         "Stepsize and covariance tuner, 19 samples, regularization 7.0"
#     b = bracketed_doubling_tuner() # testing the defaults
#     @test b isa TunerSequence
#     @test b.tuners == (StepsizeTuner(75), # init
#                        StepsizeCovTuner(25, 5.0), # doubling each step
#                        StepsizeCovTuner(50, 5.0),
#                        StepsizeCovTuner(100, 5.0),
#                        StepsizeCovTuner(200, 5.0),
#                        StepsizeCovTuner(400, 5.0),
#                        StepsizeTuner(50)) # terminate
#     @test repr(b) ==
#         """
# Sequence of 7 tuners, 900 total samples
#   Stepsize tuner, 75 samples
#   Stepsize and covariance tuner, 25 samples, regularization 5.0
#   Stepsize and covariance tuner, 50 samples, regularization 5.0
#   Stepsize and covariance tuner, 100 samples, regularization 5.0
#   Stepsize and covariance tuner, 200 samples, regularization 5.0
#   Stepsize and covariance tuner, 400 samples, regularization 5.0
#   Stepsize tuner, 50 samples"""
# end
