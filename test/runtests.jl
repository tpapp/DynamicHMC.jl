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
    EvaluatedLogDensity, PhasePoint, logdensity, phasepoint, rand_phasepoint, leapfrog,
    logdensity,
    # building blocks
    rand_bool, TurnStatistic, DivergenceStatistic, divergence_statistic,
    get_acceptance_rate, Trajectory,
    # stepsize
    InitialStepsizeSearch, find_initial_stepsize,
    # transitions and tuning
    transition_NUTS, NUTS_init, StepsizeTuner, StepsizeCovTuner, tune,
    TunerSequence, bracketed_doubling_tuner

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
@include_testset("test-buildingblocks.jl")
@include_testset("test-stepsize.jl")
@include_testset("test-tuners.jl")
@include_testset("test-sample-normal.jl")
@include_testset("test-normal-mcmc.jl")
@include_testset("test-statistics.jl")
@include_testset("test-reporting.jl")
