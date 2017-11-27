import DynamicHMC: rand_bool

@testset "random booleans" begin
    @test abs(mean(rand_bool(RNG, 0.3) for _ in 1:10000) - 0.3) ≤ 0.01
end
