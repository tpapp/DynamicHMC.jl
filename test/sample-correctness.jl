#####
##### sample correctness tests
#####

isinteractive() && include("common.jl")

####
#### LogDensityTestSuite is under active development, use the latest
#### FIXME remove code below when that package stabilizies and use Project.toml
####

try
    using LogDensityTestSuite
catch
    @info "installing LogDensityTestSuite"
    import Pkg
    Pkg.API.add(Pkg.PackageSpec(; url = "https://github.com/tpapp/LogDensityTestSuite.jl"))
    using LogDensityTestSuite
end

"Random unitary matrix."
rand_Q(K) = qr(randn(K, K)).Q

"Random (positive) diagonal matrix."
rand_D(K) = Diagonal(abs.(randn(K)))

function run_chains(rng, ℓ, N, K)
    [position_matrix(mcmc_with_warmup(rng, ℓ, N; reporter = NoProgressReport()).chain)
     for i in 1:K]
end

function mcmc_statistics(position_matrices)
    K = size(first(position_matrices), 1)
    R̂ = [potential_scale_reduction([mx[i, :] for mx in position_matrices]...) for i in 1:K]
    τ = mean([vec(mapslices(first ∘ ess_factor_estimate, mx; dims = 2))
              for mx in position_matrices])
    (R̂ = R̂, τ = τ)
end

"Multivariate normal with Σ = Q*D*D*Q′."
function multivariate_normal(μ, D, Q)
    shift(linear(StandardMultivariateNormal(length(μ)), Q * D), μ)
end

function NUTS_tests(rng, ℓ, N; K = 3, max_R̂ = 1.05, min_τ = 0.1,
                    KS_p̄ = 0.05, AD_p̄ = 0.05)
    mxs = run_chains(RNG, ℓ, N, K)

    # mixing diagnostics
    @unpack R̂, τ = mcmc_statistics(mxs)
    @test all(R̂ .≤ max_R̂)
    @test quantile(τ, 0.2) ≥ min_τ

    # distribution comparison tests
    Z = reduce(hcat, mxs)
    Z′ = samples(ℓ, 1000)
    d = dimension(ℓ)
    KS_stats = [ApproximateTwoSampleKSTest(Z[i, :], Z′[i, :]) for i in 1:d]
    @test minimum(pvalue, KS_stats) ≥ 0.05 / d
    AD_stats = [KSampleADTest(Z[i, :], Z′[i, :]) for i in 1:d]
    @test mean(pvalue, AD_stats) ≥ 0.05 / d
end

@testset "NUTS tests with random normal" begin
    for _ in 1:10
        K = rand(3:10)
        μ = randn(K)
        D = rand_D(K)
        Q = rand_Q(K)
        ℓ = multivariate_normal(μ, D, Q)
        NUTS_tests(RNG, ℓ, 1000)
    end
end

@testset "NUTS tests with specifically picked normals" begin
    # huge variance
    ℓ = multivariate_normal([0.0], fill(5e8, 1, 1), I)
    NUTS_tests(RNG, ℓ, 1000)

    # huge variance, offset
    ℓ = multivariate_normal([1.0], fill(5e8, 1, 1), I)
    NUTS_tests(RNG, ℓ, 1000)

    # tiny variance, offset
    ℓ = multivariate_normal([1.0], fill(5e-8, 1, 1), I)
    NUTS_tests(RNG, ℓ, 1000)
end
