"""
    $SIGNATURES

Random boolean which is `true` with the given probability `prob`.

All random numbers in this library are obtained from this function.
"""
rand_bool(rng::AbstractRNG, prob::T) where {T <: AbstractFloat} =
    rand(rng, T) ≤ prob
