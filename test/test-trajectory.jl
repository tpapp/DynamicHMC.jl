import DynamicHMC:
    leaf_stats, NUTSTerminationTest, NUTSEuclideanStats, isterminating,
    AcceptanceTuner, AcceptanceStats, PhasePoint, ⊔, acceptance_rate

@testset "NUTS termination" begin
    H = Hamiltonian(normal_density(fill(0.0, 2), I), GaussianKE(Diagonal(ones(2))))
    pp(q, sign) = PhasePoint(q, [sign, sign])
    q₀ = [0,0]
    q₁ = [1,0]
    z₀₊ = pp(q₀, 1)
    z₁₊ = pp(q₁, 1)
    z₀₋ = pp(q₀, -1)
    z₁₋ = pp(q₁, -1)
    x0 = leaf_stats(H, NUTSTerminationTest(), z₀₊)
    x1 = leaf_stats(H, NUTSTerminationTest(), z₁₊)
    x01 = x0 ⊔ x1
    @test x0 === x1 === x01 === NUTSEuclideanStats()
    @test !isterminating(x0, z₀₊, z₁₊)
    @test !isterminating(x0, z₀₋, z₁₊)
    @test !isterminating(x0, z₀₊, z₁₋)
    @test isterminating(x0, z₀₋, z₁₋)
end

@testset "acceptance rate" begin
    a(Δ) = leaf_stats(AcceptanceTuner(), Δ)
    x = a(log(0.3))
    @test isa(x, AcceptanceStats)
    @test acceptance_rate(x) ≈ 0.3
    y = a(log(0.6))
    @test isa(y, AcceptanceStats)
    @test acceptance_rate(y) ≈ 0.6
    z = x ⊔ x ⊔ y
    @test isa(z, AcceptanceStats)
    @test acceptance_rate(z) ≈ 0.4
end
