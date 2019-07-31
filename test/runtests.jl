using DynamicHMC, Test, ArgCheck, DataStructures, Distributions, DocStringExtensions,
    LinearAlgebra, MCMCDiagnostics, Parameters, Random, StatsBase, StatsFuns, Statistics,
    Suppressor

import ForwardDiff, Random

using DynamicHMC:
    # trees
    Directions, next_direction, biased_progressive_logprob2, adjacent_tree,
    sample_trajectory,
    # Hamiltonian
    GaussianKineticEnergy, kinetic_energy, ∇kinetic_energy, rand_p, Hamiltonian,
    EvaluatedLogDensity, evaluate_ℓ, PhasePoint, logdensity, leapfrog,
    logdensity,
    # NUTS
    TrajectoryNUTS, rand_bool, TurnStatistic, DivergenceStatistic, divergence_statistic,
    acceptance_rate, TreeStatisticsNUTS, Termination,
    # stepsize
    InitialStepsizeSearch, find_initial_stepsize,
    # mcmc
    # diagnostics
    ACCEPTANCE_QUANTILES

import DynamicHMC:
    # trees
    move, is_turning, combine_turn_statistics, is_divergent,
    combine_divergence_statistics, calculate_logprob2, combine_proposals, leaf

import LogDensityProblems: logdensity_and_gradient, dimension, capabilities, LogDensityProblems

include("utilities.jl")

macro include_testset(filename)
    @assert filename isa AbstractString
    quote
        @testset $(filename) begin
            include($(filename))
        end
    end
end

@include_testset("test-trees.jl")
@include_testset("test-hamiltonian.jl")
@include_testset("test-NUTS.jl")
@include_testset("test-stepsize.jl")
@include_testset("test-mcmc.jl")
# @include_testset("test-sample-normal.jl")
# @include_testset("test-normal-mcmc.jl")
@include_testset("test-diagnostics.jl")
# @include_testset("test-reporting.jl") FIXME being rewritten
