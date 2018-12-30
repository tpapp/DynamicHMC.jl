#####
##### statistics and diagnostics
#####

export EBFMI, NUTS_statistics

"""
    EBFMI(sample)

Energy Bayesian fraction of missing information. Useful for diagnosing poorly
chosen kinetic energies.

Low values (`≤ 0.3`) are considered problematic. See Betancourt (2016).
"""
EBFMI(sample) = (πs = get_neg_energy.(sample); mean(abs2, diff(πs)) / var(πs))


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
    NUTS_statistics(sample)

Return statistics about the sample (ie not the variables). Mostly useful for
NUTS diagnostics.
"""
function NUTS_statistics(sample)
    as = get_acceptance_rate.(sample)
    NUTS_Statistics(length(sample),
                    mean(as), quantile(as, ACCEPTANCE_QUANTILES),
                    counter(get_termination.(sample)), counter(get_depth.(sample)))
end

function show(io::IO, stats::NUTS_Statistics)
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
