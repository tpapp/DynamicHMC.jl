isinteractive() && include("common.jl")

#####
##### sample correctness tests
#####

"Random unitary matrix."
rand_Q(K) = qr(randn(K, K)).Q

"Random (positive) diagonal matrix."
rand_D(K) = Diagonal(abs.(randn(K)))

"""
$(SIGNATURES)

Run `K` chains of MCMC on `ℓ`, each for `N` samples, return a vector of position matrices.
"""
function run_chains(rng, ℓ, N, K)
    [position_matrix(mcmc_with_warmup(rng, ℓ, N; reporter = NoProgressReport()).chain)
     for i in 1:K]
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

"Multivariate normal with Σ = Q*D*D*Q′."
function multivariate_normal(μ, D, Q)
    shift(μ, linear(Q * D, StandardMultivariateNormal(length(μ))))
end

"""
$(SIGNATURES)

Run MCMC on `ℓ`, obtaining `N` samples from `K` independently adapted chains.

`R̂`, `τ`, and two-sided `p` statistics comparing to `B` uniform bins are obtained and
compared to thresholds that either *alert* or *fail*. The latter should be lax because of
false positives, the tests are rather hair-trigger.

Output is sent to `io`. Specifically, `title` is printed for the first alert.
"""
function NUTS_tests(rng, ℓ, title, N; K = 3, B = 10, io = stdout,
                    R̂_alert = 1.05, R̂_fail = 1.1,
                    τ_alert = 0.2, τ_fail = 0.1,
                    p_alert = 0.001, p_fail = p_alert * 0.1)
    @argcheck 1 < R̂_alert ≤ R̂_fail
    @argcheck 0 < τ_fail ≤ τ_alert
    @argcheck 0 < p_fail ≤ p_alert

    d = dimension(ℓ)

    title_printed = false
    function _print_title_once()
        if !title_printed
            println(io, "INFO while testing: $(title), dimension $(d)")
            title_printed = true
        end
    end
    mxs = run_chains(RNG, ℓ, N, K)

    # mixing and autocorrelation diagnostics
    @unpack R̂, τ = mcmc_statistics(mxs)
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

    # distribution comparison tests
    Z = reduce(hcat, mxs)
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

@testset "NUTS tests with random normal" begin
    for _ in 1:10
        K = rand(3:10)
        μ = randn(K)
        D = rand_D(K)
        Q = rand_Q(K)
        ℓ = multivariate_normal(μ, D, Q)
        title = "multivariate normal μ = $(μ) D = $(D) Q = $(Q)"
        NUTS_tests(RNG, ℓ, title, 1000; p_alert = 1e-4)
    end
end

@testset "NUTS tests with specific distributions" begin
    ℓ = multivariate_normal([0.0], fill(5e8, 1, 1), I)
    NUTS_tests(RNG, ℓ, "univariate huge variance", 1000)

    ℓ = multivariate_normal([1.0], fill(5e8, 1, 1), I)
    NUTS_tests(RNG, ℓ, "univariate huge variance, offset", 1000)

    ℓ = multivariate_normal([1.0], fill(5e-8, 1, 1), I)
    NUTS_tests(RNG, ℓ, "univariate tiny variance, offset", 1000)
end
