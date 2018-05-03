"""
    $SIGNATURES

Random boolean which is `true` with the given probability `prob`.

All random numbers in this library are obtained from this function.
"""
rand_bool{T <: AbstractFloat}(rng::AbstractRNG, prob::T) = rand(rng, T) ≤ prob
