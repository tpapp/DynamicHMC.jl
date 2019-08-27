#####
##### utilities for testing sample correctness
#####

"""
$(SIGNATURES)

Run `K` chains of MCMC on `ℓ`, each for `N` samples, return a vector of position matrices
and EBFMI statistics as fields of a `NamedTuple`.

Keyword arguments are passed to `mcmc_with_warmup`.
"""
function run_chains(rng, ℓ, N, K; mcmc_args...)
    results = [mcmc_with_warmup(rng, ℓ, N; reporter = NoProgressReport(), mcmc_args...)
               for i in 1:K]
    (position_matrices = map(r -> position_matrix(r.chain), results),
     EBFMIs = map(r -> EBFMI(r.tree_statistics), results))
end

"""
$(SIGNATURES)

`R̂` (within/between variance) and `τ` (effective sample size coefficient) statistics for
position matrices, eg the output of `run_chains`.
"""
function mcmc_statistics(position_matrices)
    K = size(first(position_matrices), 1)
    R̂ = [potential_scale_reduction([mx[i, :] for mx in position_matrices]...) for i in 1:K]
    τ = mean([vec(mapslices(first ∘ ess_factor_estimate, mx; dims = 2))
              for mx in position_matrices])
    (R̂ = R̂, τ = τ)
end

"""
$(SIGNATURES)

Run MCMC on `ℓ`, obtaining `N` samples from `K` independently adapted chains.

`R̂`, `τ`, two-sided `p` statistics comparing to `B` uniform bins, and EBFMIs are obtained
and compared to thresholds that either *alert* or *fail*. The latter should be lax because
of false positives, the tests are rather hair-trigger.

Output is sent to `io`. Specifically, `title` is printed for the first alert.

`mcmc_args` are passed down to `mcmc_with_warmup`.
"""
function NUTS_tests(rng, ℓ, title, N; K = 3, B = 10, io = stdout,
                    R̂_alert = 1.05, R̂_fail = 2 * (R̂_alert - 1) + 1,
                    τ_alert = 0.2, τ_fail = τ_alert * 0.5,
                    p_alert = 0.001, p_fail = p_alert * 0.1,
                    EBFMI_alert = 0.5, EBFMI_fail = 0.2, mcmc_args = NamedTuple())
    @argcheck 1 < R̂_alert ≤ R̂_fail
    @argcheck 0 < τ_fail ≤ τ_alert
    @argcheck 0 < p_fail ≤ p_alert
    @argcheck 0 < EBFMI_fail < EBFMI_alert

    d = dimension(ℓ)

    title_printed = false
    function _print_title_once()
        if !title_printed
            println(io, "INFO while testing: $(title), dimension $(d)")
            title_printed = true
        end
    end
    @unpack position_matrices, EBFMIs = run_chains(RNG, ℓ, N, K; mcmc_args...)

    # mixing and autocorrelation diagnostics
    @unpack R̂, τ = mcmc_statistics(position_matrices)
    max_R̂ = maximum(R̂)
    if max_R̂ > R̂_alert
        _print_title_once()
        println(io, "ALERT max R̂ = $(max_R̂)\n  R̂ = $(round.(R̂, sigdigits = 3))" )
    end
    @test all(max_R̂ ≤ R̂_fail)

    min_τ = minimum(τ)
    if min_τ < τ_alert
        _print_title_once()
        println(io, "ALERT min τ = $(min_τ)\n  τ = $(round.(τ, sigdigits = 3))" )
    end
    @test all(min_τ ≥ τ_fail)

    min_EBFMI = minimum(EBFMIs)
    if min_EBFMI < EBFMI_alert
        _print_title_once()
        println(io, "ALERT min EBFMI = $(min_EBFMI)\n  EBFMI = $(round.(EBFMIs, sigdigits = 3))" )
    end
    @test all(min_EBFMI ≥ EBFMI_fail)

    # distribution comparison tests
    Z = reduce(hcat, position_matrices)
    Z′ = samples(ℓ, 1000)
    pd_alert = p_alert / (d * B)
    pd_fail = p_fail / (d * B)
    for i in 1:d
        q = quantile_boundaries(Z′[i, :], B)
        bc = bin_counts(q, Z[i, :])
        min_p = minimum(two_sided_pvalues(bc))
        if min_p ≤ pd_alert
            _print_title_once()
            println(io, "ALERT extreme p ≈ $(round(min_p, sigdigits = 3)) for coordinate $(i) of $(d)")
            print_ascii_plot(io, bc)
            println(io)
        end
        @test min_p ≥ pd_fail
    end
end
