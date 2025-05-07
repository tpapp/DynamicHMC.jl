#####
##### Stepsize and adaptation tests
#####

using DynamicHMC: find_initial_stepsize, InitialStepsizeSearch, DualAveraging,
    initial_adaptation_state, adapt_stepsize, current_ϵ, final_ϵ, FixedStepsize,
    local_log_acceptance_ratio

@testset "stepsize general rootfinding" begin
    A = ϵ -> -ϵ*3.0
    params = InitialStepsizeSearch()
    # parameters consistency
    @test_throws ArgumentError InitialStepsizeSearch(; log_threshold = NaN) # not finite
    @test_throws ArgumentError InitialStepsizeSearch(; log_threshold = 1.0) # too large
    @test_throws ArgumentError InitialStepsizeSearch(; initial_ϵ = -0.5) # not > 0
    @test_throws ArgumentError InitialStepsizeSearch(; maxiter_crossing = 2) # too small
    # crossing
    ϵ = find_initial_stepsize(params, A)
    @test A(ϵ) > params.log_threshold > A(params.initial_ϵ)
    let params = InitialStepsizeSearch(; initial_ϵ = 0.01)
        ϵ = find_initial_stepsize(params, A)
        @test A(ϵ) < params.log_threshold < A(params.initial_ϵ)
    end
    @test_throws DynamicHMCError find_initial_stepsize(params, ϵ -> 1) # constant
end

"""
$(SIGNATURES)

A parametric random acceptance rate that depends on the stepsize. For unit
testing acceptance rate tuning.
"""
dummy_acceptance_rate(ϵ, σ = 0.05) = min(1/ϵ * exp(randn()*σ - σ^2/2), 1)

mean_dummy_acceptance_rate(ϵ, σ = 0.05) = mean(dummy_acceptance_rate(ϵ, σ) for _ in 1:10000)

@testset "dual averaging far" begin
    ϵ₀ = 100.0                # way off
    δ = 0.65
    dual_averaging = DualAveraging(; δ = δ)
    A = initial_adaptation_state(dual_averaging, ϵ₀)
    @test A.logϵ̄ == 0           # ϵ̄₀ = 1 in Gelman and Hoffman (2014)
    @test A.m == 1
    @test A.H̄ == 0
    for _ in 1:500
        A = adapt_stepsize(dual_averaging, A, dummy_acceptance_rate(current_ϵ(A)))
    end
    @test mean_dummy_acceptance_rate(final_ϵ(A)) ≈ δ atol = 0.02
end

@testset "dual averaging close" begin
    ϵ₀ = 2.0
    δ = 0.65
    dual_averaging = DualAveraging(; δ = δ)
    A = initial_adaptation_state(dual_averaging, ϵ₀)
    for _ in 1:2000
        A = adapt_stepsize(dual_averaging, A, dummy_acceptance_rate(current_ϵ(A)))
    end
    @test mean_dummy_acceptance_rate(final_ϵ(A)) ≈ δ atol = 0.01
end

@testset "dual averaging far and noisy" begin
    ϵ₀ = 20.0
    δ = 0.65
    dual_averaging = DualAveraging(; δ = δ)
    A = initial_adaptation_state(dual_averaging, ϵ₀)
    for _ in 1:2000
        A = adapt_stepsize(dual_averaging, A, dummy_acceptance_rate(current_ϵ(A), 2.0))
    end
    @test mean_dummy_acceptance_rate(final_ϵ(A), 2.0) ≈ δ atol = 0.04
end

@testset "fixed stepsize sanity checks" begin
    fs = FixedStepsize()
    ϵ = 1.0
    A = initial_adaptation_state(fs, ϵ)
    @test A == adapt_stepsize(fs, A, ϵ)
    @test current_ϵ(A) == ϵ
    @test final_ϵ(A) == ϵ
end

@testset "find reasonable stepsize - random H, z" begin
    p = InitialStepsizeSearch()
    _bkt(A, ϵ, C) = (A(ϵ) - p.log_threshold) * (A(ϵ * C) - p.log_threshold) ≤ 0
    for _ in 1:100
        (; H, z) = rand_Hz(rand(3:5))
        A = local_log_acceptance_ratio(H, z)
        ϵ = find_initial_stepsize(p, A)
        @test _bkt(A, ϵ, 0.5) || _bkt(A, ϵ, 2.0)
    end
end

@testset "error for non-finite initial density" begin
    p = InitialStepsizeSearch()
    (; H, z) = rand_Hz(2)
    z = DynamicHMC.PhasePoint(z.Q, [NaN, NaN])
    @test_throws DynamicHMCError find_initial_stepsize(p, local_log_acceptance_ratio(H, z))
end
