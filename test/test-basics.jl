import DynamicHMC: rand_bool, bracket_zero, find_zero, bracket_find_zero

@testset "random booleans" begin
    @test abs(mean(rand_bool(RNG, 0.3) for _ in 1:10000) - 0.3) ≤ 0.01
end

@testset "rootfinding" begin
    @test find_zero(identity, -50, 100, √eps()) ≈ 0 atol = √eps()
    @test bracket_find_zero(identity, 100, 1.0, 2, √eps()) ≈ 0
    @test bracket_find_zero(identity, -100, 1.0, 2, √eps()) ≈ 0
end

@testset "bracketing" begin
    x₁, fx₁, x₂, fx₂ = bracket_zero(identity, 3.0, 0.01, 1.2)
    if x₁ > x₂
        x₁, x₂ = x₂, x₁
        fx₁, fx₂ = fx₂, fx₁
    end
    @test fx₁ == x₁ < 0 < x₂ == fx₂
    @test_throws ErrorException bracket_zero(x->1.0, 1.0, 1.0, 2.0)  # no zero
    @test_throws ArgumentError bracket_zero(identity, 1.0, 0.0, 2.0) # Δ = 0
    @test_throws ArgumentError bracket_zero(identity, 1.0, 1.0, 1.0) # C = 0
end
