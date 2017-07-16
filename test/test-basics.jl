import DynamicHMC: rand_bool, find_zero, bracket_find_zero

@testset "random booleans" begin
    @test abs(mean(rand_bool(RNG, 0.3) for _ in 1:10000) - 0.3) ≤ 0.01
end

@testset "rootfinding" begin
    @test find_zero(identity, -50, 100, √eps()) ≈ 0 atol = √eps()
    @test bracket_find_zero(identity, 100, 1.0, 2, √eps()) ≈ 0
    @test bracket_find_zero(identity, -100, 1.0, 2, √eps()) ≈ 0
end
