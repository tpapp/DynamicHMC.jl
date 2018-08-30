using DynamicHMC:
    find_crossing_stepsize, bisect_stepsize, find_initial_stepsize,
    InitialStepsizeSearch,
    adapt_stepsize, DualAveragingParameters, DualAveragingAdaptation

@testset "stepsize general rootfinding" begin
    Δ = 3.0                   # shift exponential so that ϵ=1 is not in interval
    A = ϵ -> exp(-ϵ*Δ)
    invA = A -> -log(A) / Δ
    @test 0.1 ≈ invA(A(0.1)) atol = 1e-10 # inverse is OK
    params = InitialStepsizeSearch()
    @test A(params.ϵ₀) ≤ params.a_min || A(params.ϵ₀) ≥ params.a_max # outside interval
    constantA(ϵ) = params.a_max + 0.1
    # parameters and defaults
    @test params.a_min == 0.25
    @test params.a_max == 0.75
    @test params.ϵ₀ == 1.0
    @test params.C == 2.0
    @test params.maxiter_crossing == 400
    @test params.maxiter_bisect == 400
    @test_throws ArgumentError InitialStepsizeSearch(; a_min = 0.9, a_max = 0.1) # not <
    @test_throws ArgumentError InitialStepsizeSearch(; C = 0.5) # not > 1
    @test_throws ArgumentError InitialStepsizeSearch(; maxiter_crossing = 2)
    @test_throws ArgumentError InitialStepsizeSearch(; maxiter_bisect = 2)
    # crossing from below
    ϵ₀, Aϵ₀, ϵ₁, Aϵ₁ = find_crossing_stepsize(params, A, invA(params.a_min-0.1))
    @test ϵ₀ ≥ invA(params.a_min) ≥ ϵ₁
    @test Aϵ₀ ≤ params.a_min ≤ Aϵ₁
    @test_throws ErrorException find_crossing_stepsize(params, constantA, 100.0)
    # crossing from above
    ϵ₀, Aϵ₀, ϵ₁, Aϵ₁ = find_crossing_stepsize(params, A, invA(params.a_max+0.1))
    @test ϵ₀ ≤ invA(params.a_max) ≤ ϵ₁
    @test Aϵ₀ ≥ params.a_max ≥ Aϵ₁
    @test_throws ErrorException find_crossing_stepsize(params, constantA, 0.1)
    # bisection
    ϵ = bisect_stepsize(params, A, invA(params.a_max + 0.1), invA(params.a_min - 0.1))
    @test params.a_min ≤ A(ϵ) ≤ params.a_max
    @test_throws ArgumentError bisect_stepsize(params, A, 0.4, 0.3) # order
    @test_throws ArgumentError bisect_stepsize(params, A, # already in interval
                                               invA(params.a_max - 0.1),
                                               invA(params.a_min + 0.1))
    # combined algorithm - single test
    ϵ = find_initial_stepsize(params, A)
    @test params.a_min ≤ A(ϵ) ≤ params.a_max
    # combined algorithm - random tests
    for _ in 1:1000
        α = abs(randn())
        β = -abs(randn())
        A = ϵ -> exp(-(α*ϵ+β)*ϵ)  # A(0)=1, has a hump, goes to A(∞)→0
        Aϵ₀ = A(params.ϵ₀)
        # uncomment line belows for coverage debugging or inverse
        # println("α = $α, β = $β")
        # println(Aϵ₀ ≤ params.a_min ? "below" : (Aϵ₀ ≥ params.a_max ? "above" : "in"))
        # invA(A) = -(log(A)+β)/A
        ϵ = find_initial_stepsize(params, A)
        @test params.a_min ≤ A(ϵ) ≤ params.a_max
    end
end

"""
    dummy_acceptance_rate(logϵ)

A parametric random acceptance rate that depends on the stepsize. For unit
testing acceptance rate tuning.
"""
function dummy_acceptance_rate(logϵ, σ = 0.05)
    exp(-logϵ + randn()*σ - σ^2/2)  # not constrained to be ≤ 1, modify accordingly
end

@testset "dummy acceptance rate stochastic" begin
    @test mean(dummy_acceptance_rate(0.0) for _ in 1:100000) ≈ 1 atol = 0.001
    @test std(dummy_acceptance_rate(0.0, 0.05) for _ in 1:100000) ≈ 0.05 atol = 0.001
end

@testset "dual averaging far" begin
    logϵ₀ = 10.0                # way off
    δ = 0.65
    params = DualAveragingParameters(logϵ₀; δ = δ)
    A = DualAveragingAdaptation(logϵ₀)
    @test A.logϵ̄ == 0           # ϵ₀ = 0 in Gelman and Hoffman (2014)
    @test A.m == 0
    @test A.H̄ == 0
    for _ in 1:5000
        A = adapt_stepsize(params, A, min(dummy_acceptance_rate(A.logϵ), 1))
    end
    @test dummy_acceptance_rate(A.logϵ, 0) ≈ δ atol = 0.04
end

@testset "dual averaging close" begin
    logϵ₀ = 1.0                 # closer
    δ = 0.65
    params = DualAveragingParameters(logϵ₀; δ = δ)
    A = DualAveragingAdaptation(logϵ₀)
    for _ in 1:2000
        A = adapt_stepsize(params, A, min(dummy_acceptance_rate(A.logϵ), 1))
    end
    @test dummy_acceptance_rate(A.logϵ, 0) ≈ δ atol = 0.05
end

@testset "find reasonable stepsize - random H, z" begin
    p = InitialStepsizeSearch()
    for _ in 1:100
        H, z = rand_Hz(rand(3:5))
        ϵ = find_initial_stepsize(p, H, z)
        logA = neg_energy(H, leapfrog(H, z, ϵ)) - neg_energy(H, z)
        @test p.a_min ≤ exp(logA) ≤ p.a_max
    end
end

@testset "error for non-finite initial density" begin
    p = InitialStepsizeSearch()
    H = Hamiltonian(FunctionLogDensity(ValueGradient(-Inf, [0.0])), GaussianKE(1))
    z = DynamicHMC.phasepoint_in(H, [1.0], [1.0])
    @test_throws DomainError find_initial_stepsize(p, H, z)
end
