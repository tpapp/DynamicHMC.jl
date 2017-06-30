"""
Notation follows Betancourt (2017), with some differences.

"""
module DynamicHMC

"""
    ⊔(x, y)

Combine/aggregate information, and or propagate proposals from `x` and
`y`. Used internally in this library as a generic operator, not
necessarily commutative.
"""
function ⊔ end

include("utilities.jl")
include("Hamiltonian.jl")
include("trajectory.jl")

end # module
