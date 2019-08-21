isinteractive() && include("common.jl")

#####
##### sample correctness tests
#####
##### Sample from well-characterized distributions using LogDensityTestSuite, check
##### convergence and mixing, and compare.

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

@testset "NUTS tests with random normal" begin
    for _ in 1:10
        K = rand(3:10)
        μ = randn(K)
        D = rand_D(K)
        C = rand_C(K)
        ℓ = multivariate_normal(μ, D * C)
        title = "multivariate normal μ = $(μ) D = $(D) C = $(C)"
        NUTS_tests(RNG, ℓ, title, 1000; p_alert = 1e-5)
    end
end

@testset "NUTS tests with specific normal distributions" begin
    ℓ = multivariate_normal([0.0], fill(5e8, 1, 1))
    NUTS_tests(RNG, ℓ, "univariate huge variance", 1000)

    ℓ = multivariate_normal([1.0], fill(5e8, 1, 1))
    NUTS_tests(RNG, ℓ, "univariate huge variance, offset", 1000)

    ℓ = multivariate_normal([1.0], fill(5e-8, 1, 1))
    NUTS_tests(RNG, ℓ, "univariate tiny variance, offset", 1000)

    ℓ = multivariate_normal([1.0, 2.0, 3.0], Diagonal([1.0, 2.0, 3.0]))
    NUTS_tests(RNG, ℓ, "mildly scaled diagonal", 1000)

    # these tests are kept because they did produce errors for some code that turned out to
    # be buggy in the early development version; this does not meant that they are
    # particularly powerful or sensitive ones
    ℓ = multivariate_normal([-0.37833073009094703, -0.3973395239297558],
                            cholesky([0.08108928067723374 -0.19742780267879112;
                                      -0.19742780267879112 1.2886298811010262]).L)
    NUTS_tests(RNG, ℓ, "kept 2 dim", 1000)

    ℓ = multivariate_normal(
        [-1.0960316317778482, -0.2779143641884689, -0.4566289703243874],
        cholesky([2.2367476976202463 1.4710084974801891 2.41285525745893;
                  1.4710084974801891 1.1684361535929932 0.9632367554302268;
                  2.41285525745893 0.9632367554302268 4.5595606374865785]).L)
    NUTS_tests(RNG, ℓ, "kept 3 dim", 1000)

    ℓ = multivariate_normal(
        [-1.42646, 0.94423, 0.852379, -1.12906, 0.0868619, 0.948781, -0.875067, 1.07243],
        cholesky([14.8357 2.42526 -2.97011 2.08363 -1.67358 4.02846 5.57947 7.28634;
                   2.42526 10.8874 -1.08992 1.99358 1.85011 -2.29754 -0.0540131 1.79718;
                   -2.97011 -1.08992 3.05794 0.0321187 1.8052 -1.5309 1.78163 -0.0821483;
                   2.08363 1.99358 0.0321187 2.38112 -0.252784 0.666474 1.73862 2.55874;
                   -1.67358 1.85011 1.8052 -0.252784 12.3109 -2.3913 -2.99741 -1.95031;
                   4.02846 -2.29754 -1.5309 0.666474 -2.3913 4.89957 3.6118 5.22626;
                   5.57947 -0.0540131 1.78163 1.73862 -2.99741 3.6118 10.215 9.60671;
                   7.28634 1.79718 -0.0821483 2.55874 -1.95031 5.22626 9.60671 11.5554]).L)
    NUTS_tests(RNG, ℓ, "kept 8 dim", 1000)
end

@testset "NUTS tests with mixtures" begin
    ℓ1 = multivariate_normal(zeros(3), 1.0)
    D2 = Diagonal(fill(0.4, 3))
    C2 = [1.0 -0.48058358598852935 0.39971148270854306;
          0.0 0.876948924897229 -0.5361348433365906;
          0.0 0.0 0.7434985947205197]
    ℓ2 = multivariate_normal(ones(3), D2 * C2)
    ℓ = mix(0.2, ℓ1, ℓ2)
    NUTS_tests(RNG, ℓ, "mixture of two normals", 1000)
end
