using DynamicHMC, Test

# using DynamicHMC:
#     # Hamiltonian
#     GaussianKE, Hamiltonian, PhasePoint, neg_energy, phasepoint_in,
#     rand_phasepoint, leapfrog, move,
#     # building blocks
#     rand_bool, TurnStatistic, combine_turnstats, Proposal,
#     combined_logprob_logweight, combine_proposals,
#     DivergenceStatistic, combine_divstats, divergence_statistic, isdivergent,
#     get_acceptance_rate, isturning, adjacent_tree, sample_trajectory,
#     # stepsize
#     InitialStepsizeSearch, find_initial_stepsize,
#     # transitions and tuning
#     NUTS_transition, NUTS_init, StepsizeTuner, StepsizeCovTuner, tune,
#     TunerSequence, bracketed_doubling_tuner

# using Test

# using ArgCheck: @argcheck
# using DataStructures
# using Distributions
# using DocStringExtensions: SIGNATURES
# import ForwardDiff
# using LinearAlgebra
# using LogDensityProblems:
#     logdensity, dimension, ValueGradient, AbstractLogDensityProblem, LogDensityProblems
# using MCMCDiagnostics: effective_sample_size, potential_scale_reduction
# using Parameters
# import Random
# using Random: randn, rand
# using StatsBase: mean_and_cov, mean_and_std
# using StatsFuns: logaddexp
# using Statistics: mean, quantile, Statistics
# using Suppressor

include("utilities.jl")

macro include_testset(filename)
    @assert filename isa AbstractString
    quote
        @testset $(filename) begin
            include($(filename))
        end
    end
end

# @include_testset("test-Hamiltonian-leapfrog.jl")
# @include_testset("test-buildingblocks.jl")
# @include_testset("test-stepsize.jl")
# @include_testset("test-sample-dummy.jl")
# @include_testset("test-tuners.jl")
# @include_testset("test-sample-normal.jl")
# @include_testset("test-normal-mcmc.jl")
# @include_testset("test-statistics.jl")
# @include_testset("test-reporting.jl")

# NEW CODE

@include_testset("test_trajectories.jl")
