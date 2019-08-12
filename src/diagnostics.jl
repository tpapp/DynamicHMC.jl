#####
##### statistics and diagnostics
#####

module Diagnostics

export EBFMI, summarize_tree_statistics, explore_log_acceptance_ratios, leapfrog_trajectory,
    InvalidTree, REACHED_MAX_DEPTH, is_divergent

using DynamicHMC: GaussianKineticEnergy, Hamiltonian, evaluate_ℓ, InvalidTree,
    REACHED_MAX_DEPTH, is_divergent, log_acceptance_ratio, PhasePoint, rand_p, leapfrog,
    logdensity, MAX_DIRECTIONS_DEPTH

using ArgCheck: @argcheck
using DocStringExtensions: SIGNATURES, TYPEDEF
using LogDensityProblems: dimension
using Parameters: @unpack
import Random
using Statistics: mean, quantile, var

"""
$(SIGNATURES)

Energy Bayesian fraction of missing information. Useful for diagnosing poorly
chosen kinetic energies.

Low values (`≤ 0.3`) are considered problematic. See Betancourt (2016).
"""
function EBFMI(tree_statistics)
    (πs = map(x -> x.π, tree_statistics); mean(abs2, diff(πs)) / var(πs))
end

"Acceptance quantiles for [`TreeStatisticsSummary`](@ref) diagnostic summary."
const ACCEPTANCE_QUANTILES = [0.05, 0.25, 0.5, 0.75, 0.95]

"""
$(TYPEDEF)

Storing the output of [`NUTS_statistics`](@ref) in a structured way, for pretty
printing. Currently for internal use.
"""
struct TreeStatisticsSummary{T <: Real, C <: NamedTuple}
    "Sample length."
    N::Int
    "average_acceptance"
    a_mean::T
    "acceptance quantiles"
    a_quantiles::Vector{T}
    "termination counts"
    termination_counts::C
    "depth counts (first element is for `0`)"
    depth_counts::Vector{Int}
end

"""
$(SIGNATURES)

Count termination reasons in `tree_statistics`.
"""
function count_terminations(tree_statistics)
    max_depth = 0
    divergence = 0
    turning = 0
    for tree_statistic in tree_statistics
        it = tree_statistic.termination
        if it == REACHED_MAX_DEPTH
            max_depth += 1
        elseif is_divergent(it)
            divergence += 1
        else
            turning += 1
        end
    end
    (max_depth = max_depth, divergence = divergence, turning = turning)
end

"""
$(SIGNATURES)

Count depths in tree statistics.
"""
function count_depths(tree_statistics)
    c = zeros(Int, MAX_DIRECTIONS_DEPTH + 1)
    for tree_statistic in tree_statistics
        c[tree_statistic.depth + 1] += 1
    end
    c[1:something(findlast(!iszero, c), 0)]
end

"""
$(SIGNATURES)

Summarize tree statistics. Mostly useful for NUTS diagnostics.
"""
function summarize_tree_statistics(tree_statistics)
    As = map(x -> x.acceptance_rate, tree_statistics)
    TreeStatisticsSummary(length(tree_statistics),
                          mean(As), quantile(As, ACCEPTANCE_QUANTILES),
                          count_terminations(tree_statistics),
                          count_depths(tree_statistics))
end

function Base.show(io::IO, stats::TreeStatisticsSummary)
    @unpack N, a_mean, a_quantiles, termination_counts, depth_counts = stats
    println(io, "Hamiltonian Monte Carlo sample of length $(N)")
    print(io, "  acceptance rate mean: $(round(a_mean; digits = 2)), 5/25/50/75/95%:")
    for aq in a_quantiles
        print(io, " ", round(aq; digits = 2))
    end
    println(io)
    function print_percentages(pairs)
        is_first = true
        for (key, value) in sort(collect(pairs), by = first)
            if is_first
                is_first = false
            else
                print(io, ",")
            end
            print(io, " $(key) => $(round(Int, 100*value/N))%")
        end
    end
    print(io, "  termination:")
    print_percentages(pairs(termination_counts))
    println(io)
    print(io, "  depth:")
    print_percentages(zip(axes(depth_counts, 1) .- 1, depth_counts))
end

####
#### Acceptance ratio diagnostics
####

"""
$(SIGNATURES)

From an initial position, calculate the uncapped log acceptance ratio for the given log2
stepsizes and momentums `ps`, `N` of which are generated randomly by default.
"""
function explore_log_acceptance_ratios(ℓ, q, log2ϵs;
                                       rng = Random.GLOBAL_RNG,
                                       κ = GaussianKineticEnergy(dimension(ℓ)),
                                       N = 20, ps = [rand_p(rng, κ) for _ in 1:N])
    H = Hamiltonian(κ, ℓ)
    Q = evaluate_ℓ(ℓ, q)
    [log_acceptance_ratio(H, PhasePoint(Q, p), 2.0^log2ϵ) for log2ϵ in log2ϵs, p in ps]
end

"""
$(SIGNATURES)

Calculate a leapfrog trajectory visiting `positions` relative to the starting point `q`,
with stepsize `ϵ`. `positions` has to contain `0`.

Returns a `NamedTuple` of

- `Δs`, the log density + the kinetic energy relative to position `0`,

- `zs`, which are [`PhasePoint`](@ref) objects.
"""
function leapfrog_trajectory(ℓ, q, ϵ, positions::UnitRange{<:Integer};
                             rng = Random.GLOBAL_RNG,
                             κ = GaussianKineticEnergy(dimension(ℓ)), p = rand_p(rng, κ))
    A, B = first(positions), last(positions)
    @argcheck A ≤ 0 ≤ B "Positions has to contain `0`."
    Q = evaluate_ℓ(ℓ, q)
    H = Hamiltonian(κ, ℓ)
    z₀ = PhasePoint(Q, p)
    fwd_part = let z = z₀
        [(z = leapfrog(H, z, ϵ); z) for _ in 1:B]
    end
    bwd_part = let z = z₀
        [(z = leapfrog(H, z, -ϵ); z) for _ in 1:(-A)]
    end
    zs = vcat(reverse!(bwd_part), z₀, fwd_part)
    πs = logdensity.(Ref(H), zs)
    Δs = πs .- Ref(πs[1 - A])   # relative to z₀
    (Δs = Δs, zs = zs)
end

end
