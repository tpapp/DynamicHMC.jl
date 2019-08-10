#####
##### statistics and diagnostics
#####

module Diagnostics

export EBFMI, NUTS_statistics, explore_log_acceptance_ratios, leapfrog_trajectory

using DynamicHMC: Termination, GaussianKineticEnergy, Hamiltonian, evaluate_ℓ,
    log_acceptance_ratio, PhasePoint, rand_p, leapfrog, logdensity

using ArgCheck: @argcheck
using DataStructures: counter
using DocStringExtensions: SIGNATURES
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
EBFMI(sample) = (πs = map(x -> x.π, sample); mean(abs2, diff(πs)) / var(πs))


"Acceptance quantiles for [`NUTS_Statistics`](@ref) diagnostic summary."
const ACCEPTANCE_QUANTILES = range(0; stop = 1, length = 5)

"""
Storing the output of [`NUTS_statistics`](@ref) in a structured way, for pretty
printing. Currently for internal use.
"""
struct NUTS_Statistics{T <: Real,
                       DT <: AbstractDict{Termination,Int},
                       DD <: AbstractDict{Int,Int}}
    "Sample length."
    N::Int
    "average_acceptance"
    a_mean::T
    "acceptance quantiles"
    a_quantiles::Vector{T}
    "termination counts"
    termination_counts::DT
    "depth counts"
    depth_counts::DD
end

"""
$(SIGNATURES)

Return statistics about the sample (ie not the variables). Mostly useful for
NUTS diagnostics.
"""
function NUTS_statistics(sample)
    as = map(x -> x.acceptance_statistic, sample)
    NUTS_Statistics(length(sample),
                    mean(as), quantile(as, ACCEPTANCE_QUANTILES),
                    counter(map(x -> x.termination, sample)),
                    counter(map(x -> x.depth, sample)))
end

function Base.show(io::IO, stats::NUTS_Statistics)
    @unpack N, a_mean, a_quantiles, termination_counts, depth_counts = stats
    println(io, "Hamiltonian Monte Carlo sample of length $(N)")
    print(io, "  acceptance rate mean: $(round(a_mean; digits = 2)), min/25%/median/75%/max:")
    for aq in a_quantiles
        print(io, " ", round(aq; digits = 2))
    end
    println(io)
    function print_dict(dict)
        for (key, value) in sort(collect(dict), by = first)
            print(io, " $(key) => $(round(Int, 100*value/N))%")
        end
    end
    print(io, "  termination:")
    print_dict(termination_counts)
    println(io)
    print(io, "  depth:")
    print_dict(depth_counts)
    println(io)
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
