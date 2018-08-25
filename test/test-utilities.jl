@testset "simulated meancov" begin
    d = MvNormal([2,2], [2.0,1.0])
    m, C = simulated_meancov(()->rand(d), 10000)
    @test m ≈ mean(d) atol = 0.05 rtol = 0.1
    @test C ≈ cov(d) atol = 0.05 rtol = 0.1
end
