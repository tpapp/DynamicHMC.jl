"""
    rand_bool(prob)

Random boolean which is `true` with the given`probability `prob`.
"""
rand_bool{T <: AbstractFloat}(prob::T) = rand(T) ≤ prob

"""
    rand_transition(logprob)

`true` with probability `exp(logprob)`, `false` otherwise. Note that
`logprob > 0` is allowed and always returns `true`. Useful for
Metropolis-Hastings and similar transitions.
"""
rand_transition(logprob) = logprob ≥ 0 || rand_bool(exp(logprob))
