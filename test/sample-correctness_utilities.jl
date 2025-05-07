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
    results = OhMyThreads.tcollect(mcmc_with_warmup(rng, ℓ, N; reporter = NoProgressReport(), mcmc_args...)
                                   for _ in 1:K)
    (stacked_posterior_matrices = stack_posterior_matrices(results),
     concat_posterior_matrices = pool_posterior_matrices(results),
     EBFMIs = map(r -> EBFMI(r.tree_statistics), results))
end

###
### Multivariate normal ℓ for testing
###

"Random Cholesky factor for correlation matrix."
function rand_C(K)
    t = TransformVariables.CorrCholeskyFactor(K)
    TransformVariables.transform(t, randn(RNG, TransformVariables.dimension(t)) ./ 4)'
end

"""
$(SIGNATURES)

`R̂` (within/between variance) and `τ` (effective sample size coefficient) statistics for
posterior matrices, eg the output of `run_chains`.
"""
function mcmc_statistics(stacked_posterior_matrices)
    (; ess, rhat) = ess_rhat(stacked_posterior_matrices)
    (R̂ = rhat, τ = ess ./ size(stacked_posterior_matrices, 1))
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
function NUTS_tests(rng, ℓ, title, N; K = 5, io = stdout, mcmc_args = NamedTuple(),
                    R̂_alert = 1.01, R̂_fail = 2 * (R̂_alert - 1) + 1,
                    τ_alert = 1.0, τ_fail = τ_alert * 0.5,
                    p_alert = 0.1, p_fail = p_alert * 0.1,
                    EBFMI_alert = 0.5, EBFMI_fail = EBFMI_alert / 2)
    @argcheck 1 < R̂_alert ≤ R̂_fail
    @argcheck 0 < τ_fail ≤ τ_alert
    @argcheck 0 < p_fail ≤ p_alert
    @argcheck 0 < EBFMI_fail < EBFMI_alert

    d = dimension(ℓ)
    _round(x) = round(x; sigdigits = 3) # for printing

    title_printed = false
    function _print_diagnostics(label, is_min, value, alert_threshold, error_threshold)
        if !title_printed
            printstyled(io, "INFO while testing: $(title), dimension $(d)\n";
                        color = :blue, bold = true)
            title_printed = true
        end
        if is_min
            vm = minimum(value)
            if vm ≥ alert_threshold
                mark, rel, vt, col = '✓', '≥', alert_threshold, :green
            elseif vm ≥ error_threshold
                mark, rel, vt, col = '!', '≱', alert_threshold, :yellow
            else
                mark, rel, vt, col = '✘', '≱', error_threshold, :red
            end
        else
            vm = maximum(value)
            if vm ≤ alert_threshold
                mark, rel, vt, col = '✓', '≤', alert_threshold, :green
            elseif vm ≤ error_threshold
                mark, rel, vt, col = '!', '≰', alert_threshold, :yellow
            else
                mark, rel, vt, col = '✘', '≰', error_threshold, :red
            end
        end
        printstyled(io, "$(mark) $(label) = $(_round.(vm)) $(rel) $(_round(vt))\n";
                    color = col)
    end
    (; stacked_posterior_matrices, concat_posterior_matrices, EBFMIs) =
        run_chains(RNG, ℓ, N, K; mcmc_args...)

    # mixing and autocorrelation diagnostics
    (; R̂, τ) = mcmc_statistics(stacked_posterior_matrices)
    _print_diagnostics("R̂", false, R̂, R̂_alert, R̂_fail)
    @test all(maximum(R̂) ≤ R̂_fail)
    _print_diagnostics("τ", true, τ, τ_alert, τ_fail)
    @test all(minimum(τ) ≥ τ_fail)
    _print_diagnostics("EBFMI", true, EBFMIs, EBFMI_alert, EBFMI_fail)
    @test all(minimum(EBFMIs) ≥ EBFMI_fail)

    # distribution comparison tests
    Z = concat_posterior_matrices
    Z′ = samples(ℓ, 1000)
    pd_alert = p_alert / d      # a simple Bonferroni correction
    pd_fail = p_fail / d
    ps = map((a, b) -> pvalue(KSampleADTest(a, b)), eachrow(Z), eachrow(Z′))
    _print_diagnostics("p", true, ps, p_alert, p_fail)
    @test all(minimum(ps) ≥ p_fail)
end
