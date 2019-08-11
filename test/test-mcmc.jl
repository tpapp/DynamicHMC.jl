isinteractive() && include("common.jl")

#####
##### Test building blocks of MCMC
#####

@testset "mcmc" begin
    ℓ = DistributionLogDensity(MvNormal(ones(5), Diagonal(ones(5))))

    # defaults
    results = mcmc_with_warmup(RNG, ℓ, 10000)
    Z = DynamicHMC.position_matrix(results.chain)
    @test norm(mean(Z; dims = 2) .- ones(5), Inf) < 0.02
    @test norm(std(Z; dims = 2) .- ones(5), Inf) < 0.02
    @test mean(x -> x.acceptance_rate, results.tree_statistics) ≥ 0.8
    @test 0.5 ≤ results.ϵ ≤ 2

    # fixed stepsize
    results = mcmc_with_warmup(RNG, ℓ, 10000;
                               initialization = (ϵ = 1.0, ),
                               warmup_stages = fixed_stepsize_warmup_stages())
    Z = DynamicHMC.position_matrix(results.chain)
    @test norm(mean(Z; dims = 2) .- ones(5), Inf) < 0.02
    @test norm(std(Z; dims = 2) .- ones(5), Inf) < 0.02
    @test mean(x -> x.acceptance_rate, results.tree_statistics) ≥ 0.7
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
