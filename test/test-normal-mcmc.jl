"""
$(SIGNATURES)

Return the normalized value (x̂-x̄)/MCMCSE(x). If the sampler is correct, it follows a
Normal(0,1) distribution.

MCMCSE(x) is the MCMC standard error, corrected by the effective sample size instead of the
sample size.
"""
function zvalue(xs, x̄)
    ess = effective_sample_size(xs)
    x̂, σ = mean_and_std(xs; corrected = false)
    (x̂ - x̄) / (σ / √ess)
end

"Test structure for easier reporting."
struct ZTest{Ta, Tf}
    "Name of test statistic."
    name::String
    "Extract the value from variables."
    accessor::Ta
    "Theoretical mean."
    x̄::Tf
end

"""
$(SIGNATURES)

Calculate a `zvalue` from a `sample`.
"""
function zvalue(sample, test::ZTest)
    @unpack name, accessor, x̄ = test
    name => zvalue(map(accessor ∘ get_position, sample), x̄)
end

"""
$(SIGNATURES)

Print a warning when `|z| ≥ threshold`.
"""
function zvalue_warn(name_and_z::Pair, threshold)
    name, z = name_and_z
    if abs(z) ≥ threshold
        @warn "$(name): z = $(z)"
    end
end

"""
$(SIGNATURES)

z threshold at which the maximum `|z|` out of `M` variables has cdf `1-p`,
assuming normality.
"""
zthreshold(M, p) = √quantile(Chisq(1), (1-p)^(1/M))

"""
$(SIGNATURES)

Return mean and covariance tests for multivariate distributions.
"""
function mean_cov_ztests(dist::Distribution{Multivariate,Continuous})
    K = length(dist)
    μ = mean(dist)
    Σ = cov(dist)
    tests = Vector{ZTest}()
    for i in 1:K
        μi = μ[i]
        push!(tests, ZTest("μ($i)", x->x[i], μi))
        for j in 1:i
            μj = μ[j]
            push!(tests, ZTest("Σ($i,$j)", x->(x[i]-μi)*(x[j]-μj), Σ[i,j]))
        end
    end
    tests
end

"""
    $SIGNATURES

Run `M` chains of length `N` using `sampler`, then calculate the columnwise R̂
(potential scale reduction factor).

`sampler` is assumed to be adapted, no adaptation is performed.
"""
function R̂(sampler, N, M)
    variables = [get_position_matrix(mcmc(sampler, N)) for _ in 1:M]
    K = size(variables[1], 2)
    [potential_scale_reduction(collect(v[:, j] for v in variables)...) for j in 1:K]
end

@testset "normal z tests fixed" begin
    ℓ0 = MvNormal([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
    ℓ1 = MvNormal([-0.37833073009094703, -0.3973395239297558],
                  [0.08108928067723374 -0.19742780267879112;
                   -0.19742780267879112 1.2886298811010262])
    ℓ2 = MvNormal([-1.0960316317778482, -0.2779143641884689, -0.4566289703243874],
                  [2.2367476976202463 1.4710084974801891 2.41285525745893;
                   1.4710084974801891 1.1684361535929932 0.9632367554302268;
                   2.41285525745893 0.9632367554302268 4.5595606374865785])
    ℓ3 = MvNormal([-1.42646, 0.94423, 0.852379, -1.12906, 0.0868619, 0.948781, -0.875067, 1.07243],
                  [14.8357 2.42526 -2.97011 2.08363 -1.67358 4.02846 5.57947 7.28634;
                   2.42526 10.8874 -1.08992 1.99358 1.85011 -2.29754 -0.0540131 1.79718;
                   -2.97011 -1.08992 3.05794 0.0321187 1.8052 -1.5309 1.78163 -0.0821483;
                   2.08363 1.99358 0.0321187 2.38112 -0.252784 0.666474 1.73862 2.55874;
                   -1.67358 1.85011 1.8052 -0.252784 12.3109 -2.3913 -2.99741 -1.95031;
                   4.02846 -2.29754 -1.5309 0.666474 -2.3913 4.89957 3.6118 5.22626;
                   5.57947 -0.0540131 1.78163 1.73862 -2.99741 3.6118 10.215 9.60671;
                   7.28634 1.79718 -0.0821483 2.55874 -1.95031 5.22626 9.60671 11.5554])
    for ℓ in [ℓ0, ℓ1, ℓ2, ℓ3]
        chain, nuts = NUTS_init_tune_mcmc(RNG, DistributionLogDensity(ℓ), 1000)
        @test EBFMI(chain) ≥ 0.3
        @test maximum(R̂(nuts, 1000, 3)) ≤ 1.05
        zs = zvalue.([chain], mean_cov_ztests(ℓ))
        zvalue_warn.(zs, 4)
        @test maximum(abs ∘ last, zs) ≤ zthreshold(length(zs), 0.001)
    end
end

@testset "normal z tests random" begin
    for _ in 1:100
        K = rand(2:10)
        ℓ = MvNormal(randn(K), Matrix(rand_Σ(K)))
        chain, nuts = NUTS_init_tune_mcmc(RNG, DistributionLogDensity(ℓ), 1000;
                                           report = ReportSilent())
        @test EBFMI(chain) ≥ 0.3
        @test maximum(R̂(nuts, 1000, 3)) ≤ 1.05
        zs = zvalue.([chain], mean_cov_ztests(ℓ))
        zvalue_warn.(zs, 4)
        @test maximum(abs ∘ last, zs) ≤ zthreshold(length(zs), 0.001) + 0.9 + 0.5*RELAX
    end
end

@testset "default RNG interface" begin
    chain, nuts = NUTS_init_tune_mcmc(DistributionLogDensity(MvNormal, 5), 1000;
                                      report = ReportSilent())
    @test EBFMI(chain) ≥ 0.3
    @test repr(nuts) isa AbstractString # just test that it prints w/o error
end
