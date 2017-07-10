"""
Notation follows Betancourt (2017), with some differences.

"""
module DynamicHMC

using ArgCheck
using Parameters
import StatsFuns: logsumexp
import Base.Random: GLOBAL_RNG

export
    KineticEnergy,
    EuclideanKE,
    GaussianKE,
    logdensity,
    loggradient,
    HMCTransition,
    HMC_transition

include("Hamiltonian.jl")
include("stepsize.jl")
include("divergence.jl")
include("proposal.jl")
include("turn.jl")
include("sampling.jl")

end # module
