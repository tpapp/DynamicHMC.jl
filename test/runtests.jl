using DynamicHMC
using Base.Test

# consistent testing
const RNG = srand(UInt32[0x23ef614d, 0x8332e05c, 0x3c574111, 0x121aa2f4])

include("utilities.jl")
include("test-basics.jl")
include("test-Hamiltonian-leapfrog.jl")
include("test-buildingblocks.jl")
include("test-stepsize.jl")
# include("test-sample-dummy.jl")
