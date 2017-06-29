using DynamicHMC

using Base.Test

# consistent testing
srand(UInt32[0x23ef614d, 0x8332e05c, 0x3c574111, 0x121aa2f4])

include("test-utilities.jl")
include("test-Hamiltonian.jl")
