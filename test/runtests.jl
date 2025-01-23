using DynamicHMC, Test, ArgCheck, DocStringExtensions, HypothesisTests, LinearAlgebra,
    Random, Statistics
using LogExpFunctions: logaddexp, log1mexp
using StatsBase: mean_and_cov
using Logging: with_logger, NullLogger
import ForwardDiff, Random, TransformVariables
using DynamicHMC.Diagnostics
using DynamicHMC.Diagnostics: ACCEPTANCE_QUANTILES
using LogDensityProblems: logdensity_and_gradient, dimension, LogDensityProblems
using LogDensityTestSuite

###
### general test environment
###

const RNG = Random.default_rng()   # shorthand
Random.seed!(RNG, UInt32[0x23ef614d, 0x8332e05c, 0x3c574111, 0x121aa2f4])

"Tolerant testing in a CI environment."
const RELAX = (k = "CONTINUOUS_INTEGRATION"; haskey(ENV, k) && ENV[k] == "true")

include("utilities.jl")

####
#### unit tests
####

include("test_trees.jl")
include("test_hamiltonian.jl")
include("test_NUTS.jl")
include("test_stepsize.jl")
include("test_mcmc.jl")
include("test_diagnostics.jl")
include("test_logging.jl")

####
#### sample correctness tests
####

include("sample-correctness_tests.jl")

####
#### static analysis and QA
####

# do not test on older Julia versions and nightly
if VERSION >= v"1.7" && isempty(VERSION.prerelease)
    include("jet.jl")
end

@testset "Aqua" begin
    import Aqua
    Aqua.test_all(DynamicHMC; ambiguities = false)
    # testing separately, cf https://github.com/JuliaTesting/Aqua.jl/issues/77
    Aqua.test_ambiguities(DynamicHMC)
end
