"""
    leaf_stats(X, ...)

Create statistics from single-point trajectory ("leaf"), which can
then be aggregated with `⊔`.
"""
function leaf_stats end

"""
Termination test for trajectories.

Subtypes support the following interface:

`leaf_termination_stats` are used to generate termination statistics
for single points. `⊔` is then used to combine these along a tree.

"""
abstract type TerminationTest end

"""
Termination criteria for NUTS. Since there are no parameters, this is
just a singleton type.
"""
struct NUTSTerminationTest <: TerminationTest end

"For an Euclidean metric, we collect no information, and use the
termination statistics in the NUTS paper (Gelman and Hoffman 2014)."
struct NUTSEuclideanStats end

leaf_stats(::Hamiltonian{<: Any, <: EuclideanKE}, ::NUTSTerminationTest, z) =
    NUTSEuclideanStats()

⊔(::NUTSEuclideanStats, ::NUTSEuclideanStats) = NUTSEuclideanStats()

function isterminating(::NUTSEuclideanStats, z₋, z₊)
    dq = z₊.q - z₋.q
    term(z) = dot(z.p, dq) < 0
    term(z₋) && term(z₊)
end

"""
Statistics about a trajectory, for all visited phase points.

Note that this set is broader than the points that are sampled from,
since it also contains information for subtrees which were divergent
or terminating.
"""
struct AcceptanceTuner end

"""
This information is mainly used for tuning. See Gelman and Hoffman
(2014), Section 3.2. FIXME add ref for tuner
"""
struct AcceptanceStats{T <: Real}
    "Sum of min(π(z)-π₀, 0) for all phase points."
    total::T
    "Number of leapfrog steps."
    steps::Int
end

⊔(x::AcceptanceStats, y::AcceptanceStats) =
    AcceptanceStats(x.total + y.total, x.steps + y.steps)

leaf_stats(::AcceptanceTuner, Δ) = AcceptanceStats(Δ > 0 ? one(Δ) : exp(Δ), 1)

acceptance_rate(as::AcceptanceStats) = as.total / as.steps
