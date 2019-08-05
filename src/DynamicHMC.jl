"""
Implementation of the No U-Turn Sampler for MCMC.

Please read the documentation. For the impatient: you probably want to

1. define a log density problem (eg for Bayesian inference) using the `LogDensityProblems`
package, then

2. use it with [`mcmc_with_warmup`](@ref).
"""
module DynamicHMC

export
    # kinetic energy
    GaussianKineticEnergy,
    # NUTS
    TreeStatisticsNUTS,
    # mcmc
    InitialStepsizeSearch, DualAveragingAdaptation, WarmupState, FindLocalOptimum,
    TuningNUTS, mcmc_with_warmup,
    # diagnostics
    EBFMI, NUTS_statistics, explore_local_acceptance_ratios

using ArgCheck: @argcheck
using DataStructures: counter
using DocStringExtensions: FIELDS, FUNCTIONNAME, SIGNATURES, TYPEDEF
using LinearAlgebra
using LinearAlgebra: checksquare, Diagonal, Symmetric
using LogDensityProblems: capabilities, LogDensityOrder, dimension, logdensity_and_gradient
import NLSolversBase, Optim # optimization step in mcmc
using Parameters: @unpack
using Random: AbstractRNG, randn, Random
using Statistics: cov, mean, median, middle, quantile, var
using StatsFuns: logaddexp

include("trees.jl")
include("hamiltonian.jl")
include("stepsize.jl")
include("NUTS.jl")
include("reporting.jl")
include("mcmc.jl")
include("diagnostics.jl")

end # module
