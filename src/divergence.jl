struct DivergenceStatistic{Tf}
    "`true` iff the sampler was terminated because of divergence."
    divergent::Bool
    "Sum of metropolis acceptances probabilities over the whole
    trajectory (including invalid parts)."
    ∑a::Tf
    "Total number of leapfrog steps."
    steps::Int
end

DivergenceStatistic() = DivergenceStatistic(false, 0.0, 0)

isdivergent(x::DivergenceStatistic) = x.divergent

function ⊔(x::DivergenceStatistic, y::DivergenceStatistic)
    DivergenceStatistic(x.divergent || y.divergent, x.∑a + y.∑a, x.steps + y.steps)
end

acceptance_rate(x::DivergenceStatistic) = x.∑a / x.steps
