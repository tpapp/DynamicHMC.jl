isinteractive() && include("common.jl")
include("sample-correctness-utilities.jl")

#####
##### sample correctness tests
#####
##### Sample from well-characterized distributions using LogDensityTestSuite, check
##### convergence and mixing, and compare.

@testset "NUTS tests with random normal" begin
    for _ in 1:10
        K = rand(3:10)
        μ = randn(K)
        D = rand_D(K)
        C = rand_C(K)
        ℓ = multivariate_normal(μ, D * C)
        title = "multivariate normal μ = $(μ) D = $(D) C = $(C)"
        NUTS_tests(RNG, ℓ, title, 1000)
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

@testset "NUTS tests with heavier tails and skewness" begin
    K = 5

    ℓ = elongate(1.2, StandardMultivariateNormal(K))
    NUTS_tests(RNG, ℓ, "elongate(1.2, 𝑁)", 1000; p_alert = 1e-5, EBFMI_alert = 0.2)

    ℓ = elongate(1.1, shift(ones(K), StandardMultivariateNormal(K)))
    NUTS_tests(RNG, ℓ, "skew elongate(1.1, 𝑁)", 1000)
end
