"""
Statistics for the identification of turning points. See Betancourt (2017).

Note that the ± subscripts of p♯ indicate direction of expansion.
"""
struct TurnStatistic{T}
    p♯₋::T
    p♯₊::T
    ρ::T
end

"""
Combine turn statistics of two (adjacent) trajectories.
"""
⊔(x::TurnStatistic, y::TurnStatistic) = TurnStatistic(x.p♯₋, y.p♯₊, x.ρ + y.ρ)

"""
Test termination based on turn statistics. Uses the generalized NUTS
criterion from Betancourt (2017).
"""
function isturning(stats::TurnStatistic)
    @unpack p♯₋, p♯₊, ρ = stats
    dot(p♯₋, ρ) < 0 || dot(p♯₊, ρ) < 0
end
