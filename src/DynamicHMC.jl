__precompile__()
module DynamicHMC

import Base: rand, length, show
import Base.LinAlg.checksquare

using ArgCheck: @argcheck
import Compat                   # for DomainError(val, msg) in v0.6
using DataStructures: counter
using DiffResults: value, gradient
using DocStringExtensions: SIGNATURES, FIELDS
using Parameters: @unpack
import StatsFuns: logsumexp

include("utilities.jl")
include("hamiltonian.jl")
include("stepsize.jl")
include("buildingblocks.jl")
include("sampler.jl")
include("diagnostics.jl")

end # module
