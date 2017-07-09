import DynamicHMC:
    TurnStatistic, isturning,
    combine_proposals, ProposalPoint, ⊔,
    DivergenceStatistic, acceptance_rate

import StatsFuns: logsumexp

@testset "low-level turn statistics" begin
    n = 3
    ρ₁ = ones(n)
    ρ₂ = 2*ρ₁
    p₁ = ρ₁
    p₂ = -ρ₁
    ts₁ = TurnStatistic(p₁, p₁, ρ₁)
    ts₂ = TurnStatistic(p₂, p₂, ρ₂)
    @test ts₁ ⊔ ts₂ ≂ TurnStatistic(p₁, p₂, ρ₁+ρ₂)
    @test !isturning(TurnStatistic(p₁, p₁, ρ₁))
    @test isturning(TurnStatistic(p₁, p₂, ρ₁))
    @test isturning(TurnStatistic(p₂, p₂, ρ₁))
    @test isturning(TurnStatistic(p₂, p₂, ρ₁))
end

@testset "low-level divergence statistics" begin
    a(p, divergent = false) = DivergenceStatistic(divergent, p, 1)
    x = a(0.3)
    @test acceptance_rate(x) ≈ 0.3
    y = a(0.6)
    @test acceptance_rate(y) ≈ 0.6
    z = x ⊔ x ⊔ y
    @test acceptance_rate(z) ≈ 0.4
end

######################################################################
# proposals
######################################################################

@testset "proposal" begin
    function test_sample(rng, prop1, prop2, bias2, prob_prob2; atol = 0.02, N = 10000)
        count = 0
        for _ in 1:N
            prop = combine_proposals(rng, prop1, prop2, bias2)
            if prop.z ≂ prop2.z
                count += 1
            else
                @test prop.z ≂ prop1.z
            end
            @test prop.logweight ≈ logsumexp(prop1.logweight, prop2.logweight)
        end
        @test count / N ≈ prob_prob2 atol = atol
    end
    prop1 = ProposalPoint(1, log(1.0))
    prop2 = ProposalPoint(2, log(3.0))
    prop3 = ProposalPoint(3, log(1/3))
    test_sample(GLOBAL_RNG, prop1, prop2, true, 1; atol = 0, N = 100)
    test_sample(GLOBAL_RNG, prop1, prop2, false, 0.75)
    test_sample(GLOBAL_RNG, prop1, prop3, true, 1/3)
    test_sample(GLOBAL_RNG, prop1, prop3, false, 0.25)
end
