import DynamicHMC: rand_bool, rand_transition

@testset "random booleans" begin
    @test abs(mean(rand_bool(0.3) for _ in 1:10000) - 0.3) ≤ 0.01
    @test abs(mean(rand_transition(log(0.3)) for _ in 1:10000) - 0.3) ≤ 0.01
end
