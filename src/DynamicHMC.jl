"""
Implementation of the No U-Turn Sampler for MCMC.

Please [read the documentation](https://tamaspapp.eu/DynamicHMC.jl/dev/). For the
impatient: you probably want to

1. define a log density problem (eg for Bayesian inference) using the `LogDensityProblems`
package, then

2. use it with [`mcmc_with_warmup`](@ref).
"""
module DynamicHMC

using ArgCheck: @argcheck
using DocStringExtensions: FIELDS, FUNCTIONNAME, SIGNATURES, TYPEDEF
using FillArrays: Fill
using LinearAlgebra: checksquare, cholesky, diag, dot, Diagonal, Symmetric, UniformScaling
using LogDensityProblems: capabilities, LogDensityOrder, dimension, logdensity_and_gradient
using LogExpFunctions: logaddexp
using Random: AbstractRNG, randn, Random, randexp
using SimpleUnPack: @unpack
using Statistics: cov, mean, median, middle, quantile, var
using TensorCast: @cast, TensorCast

include("utilities.jl")
include("trees.jl")
include("hamiltonian.jl")
include("stepsize.jl")
include("NUTS.jl")
include("reporting.jl")
include("mcmc.jl")
include("diagnostics.jl")

end # module
