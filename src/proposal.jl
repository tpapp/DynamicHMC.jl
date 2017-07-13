import StatsFuns: logsumexp

"""
    rand_bool(rng, prob)

Random boolean which is `true` with the given`probability `prob`.
"""
rand_bool{T <: AbstractFloat}(rng::AbstractRNG, prob::T) = rand(rng, T) ≤ prob

"""
Proposal that is propagated through by sampling recursively when
building the trees.
"""
struct ProposalPoint{Tz,Tf}
    "Proposed point."
    z::Tz
    "Log weight (log(∑ exp(Δ)) of trajectory/subtree)."
    logweight::Tf
end

function transition_logprob_logweight(logweight_x, logweight_y, bias_y)
    logweight = logsumexp(logweight_x, logweight_y)
    logweight_y - (bias_y ? logweight_x : logweight), logweight
end

"""
    combine_proposal(x, y, bias)

Combine proposals from two trajectories, using their weights.

When `bias_y`, biases towards `y`, introducing anti-correlations.
"""
function combine_proposals(rng, x::ProposalPoint, y::ProposalPoint, bias_y)
    logprob_y, logweight = transition_logprob_logweight(x.logweight, y.logweight, bias_y)
    z = (logprob_y ≥ 0 || rand_bool(rng, exp(logprob_y))) ? y.z : x.z
    ProposalPoint(z, logweight)
end

leaf_proposal(::Type{ProposalPoint}, z, Δ) = ProposalPoint(z, Δ)
