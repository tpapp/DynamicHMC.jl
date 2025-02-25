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
