function dummy_acceptance_rate(logϵ, σ = 0.05)
    exp(-logϵ+randn()*σ-σ^2/2)  # not constrained to be ≤ 1, modify accordingly
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
    for _ in 1:5000
        A = adapt(params, A, min(dummy_acceptance_rate(A.logϵ), 1))
    end
    @test dummy_acceptance_rate(A.logϵ, 0) ≈ δ atol = 0.03
end

@testset "dual averaging close" begin
    logϵ₀ = 1.0                 # closer
    δ = 0.65
    params = DualAveragingParameters(logϵ₀; δ = δ)
    A = DualAveragingAdaptation(logϵ₀)
    for _ in 1:2000
        A = adapt(params, A, min(dummy_acceptance_rate(A.logϵ), 1))
    end
    @test dummy_acceptance_rate(A.logϵ, 0) ≈ δ atol = 0.03
end
