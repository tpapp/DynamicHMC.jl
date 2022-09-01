#####
##### utilities for testing sample correctness
#####

using MCMCDiagnosticTools: ess_rhat

"""
$(SIGNATURES)

Run `K` chains of MCMC on `ℓ`, each for `N` samples, return a posterior matrices stacked
(indexed by `[draw, parameter, chain]`) and concatenated (indexed by `[draw, parameter]`),
and EBFMI statistics as fields of a `NamedTuple`.

Keyword arguments are passed to `mcmc_with_warmup`.
"""
function run_chains(rng, ℓ, N, K; mcmc_args...)
    results = [mcmc_with_warmup(rng, ℓ, N; reporter = NoProgressReport(), mcmc_args...)
               for i in 1:K]
    (stacked_posterior_matrices = stack_posterior_matrices(results),
     concat_posterior_matrices = pool_posterior_matrices(results),
     EBFMIs = map(r -> EBFMI(r.tree_statistics), results))
end

"""
$(SIGNATURES)

`R̂` (within/between variance) and `τ` (effective sample size coefficient) statistics for
posterior matrices, eg the output of `run_chains`.
"""
function mcmc_statistics(stacked_posterior_matrices)
    ess, R̂ = ess_rhat(stacked_posterior_matrices)
    (R̂, τ = ess ./ size(stacked_posterior_matrices, 1))
end

"""
$(SIGNATURES)

Jitter in `(-ϵ, +ϵ)`. Useful for tiebreaking for the Kolmogorov-Smirnov tests.
"""
jitter(rng, len, ϵ = 64*eps()) = (2 * ϵ) .* (rand(rng, len) .-  0.5)

"""
$(SIGNATURES)

Run MCMC on `ℓ`, obtaining `N` samples from `K` independently adapted chains.

`R̂`, `τ`, Kolmogorov-Smirnov and Anderson-Darling `p`, and EBFMIs are obtained and compared
to thresholds that either *alert* or *fail*. The latter should be lax because of false
positives, the tests can be rather hair-trigger.

Output is sent to `io`. Specifically, `title` is printed for the first alert.

`mcmc_args` are passed down to `mcmc_with_warmup`.
"""
function NUTS_tests(rng, ℓ, title, N; K = 3, io = stdout, mcmc_args = NamedTuple(),
                    R̂_alert = 1.05, R̂_fail = 2 * (R̂_alert - 1) + 1,
                    τ_alert = 0.2, τ_fail = τ_alert * 0.5,
                    p_alert = 0.001, p_fail = p_alert * 0.1,
                    EBFMI_alert = 0.5, EBFMI_fail = EBFMI_alert / 2)
    @argcheck 1 < R̂_alert ≤ R̂_fail
    @argcheck 0 < τ_fail ≤ τ_alert
    @argcheck 0 < p_fail ≤ p_alert
    @argcheck 0 < EBFMI_fail < EBFMI_alert

    d = dimension(ℓ)
    _round(x) = round(x; sigdigits = 3) # for printing

    title_printed = false
    function _print_title_once()
        if !title_printed
            printstyled(io, "INFO while testing: $(title), dimension $(d)\n";
                        color = :blue, bold = true)
            title_printed = true
        end
    end
    @unpack stacked_posterior_matrices, concat_posterior_matrices, EBFMIs =
        run_chains(RNG, ℓ, N, K; mcmc_args...)

    # mixing and autocorrelation diagnostics
    @unpack R̂, τ = mcmc_statistics(stacked_posterior_matrices)
    max_R̂ = maximum(R̂)
    if max_R̂ > R̂_alert
        _print_title_once()
        println(io, "ALERT some R̂ = $(_round.(R̂)) > $(_round(R̂_alert))" )
    end
    @test all(max_R̂ ≤ R̂_fail)

    min_τ = minimum(τ)
    if min_τ < τ_alert
        _print_title_once()
        println(io, "ALERT some τ = $(_round.(τ)) < $(_round(τ_alert))" )
    end
    @test all(min_τ ≥ τ_fail)

    min_EBFMI = minimum(EBFMIs)
    if min_EBFMI < EBFMI_alert
        _print_title_once()
        println(io, "ALERT some EBFMI = $(_round.(EBFMIs)) > $(_round(EBFMI_alert))" )
    end
    @test all(min_EBFMI ≥ EBFMI_fail)

    # distribution comparison tests
    Z = concat_posterior_matrices
    Z′ = samples(ℓ, 1000)
    pd_alert = p_alert / d      # a simple Bonferroni correction
    pd_fail = p_fail / d
    ϵ = jitter(rng, size(Z, 2))
    ϵ′ = jitter(rng, size(Z′, 2))
    for i in 1:d
        p_AD = pvalue(KSampleADTest(Z′[i, :], Z[i, :]))
        if p_AD ≤ pd_alert
            _print_title_once()
            println(io, "ALERT extreme Anderson-Darling ",
                    "p ≈ $(round(p_AD, sigdigits = 3)) < $(round(pd_alert))",
                    " for coordinate $(i) of $(d)")
        end
        @test p_AD ≥ pd_fail
    end
end
