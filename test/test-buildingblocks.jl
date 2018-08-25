
# utilities

"""
    $SIGNATURES

Recursive comparison by fields; types and values should match by `==`.
"""
function ≂(x::T, y::T) where T
    fn = fieldnames(T)
    isempty(fn) ? x == y : all(getfield(x, f) ≂ getfield(y, f) for f in fn)
end

≂(x, y) = x == y

"""
Type for testing `≂`.
"""
struct Foo{T}
    a::T
    b::T
end

@testset "≂ comparisons" begin
    @test 1 ≂ 1
    @test 1 ≂ 1.0
    @test [1,2] ≂ [1,2]
    @test [1.0,2.0] ≂ [1,2]
    @test !(1 ≂ 2)
    @test Foo(1,2) ≂ Foo(1,2)
    @test !(Foo(1,2) ≂ Foo(1,3))
    @test !(Foo{Any}(1,2) ≂ Foo(1,2))
end


# random booleans

@testset "random booleans" begin
    @test abs(mean(rand_bool(RNG, 0.3) for _ in 1:10000) - 0.3) ≤ 0.01
end


# test turn statistics

@testset "low-level turn statistics" begin
    n = 3
    ρ₁ = ones(n)
    ρ₂ = 2*ρ₁
    p₁ = ρ₁
    p₂ = -ρ₁
    τ₁ = TurnStatistic(p₁, p₁, ρ₁)
    τ₂ = TurnStatistic(p₂, p₂, ρ₂)
    @test combine_turnstats(τ₁, τ₂) ≂ TurnStatistic(p₁, p₂, ρ₁+ρ₂)
    @test !isturning(TurnStatistic(p₁, p₁, ρ₁))
    @test isturning(TurnStatistic(p₁, p₂, ρ₁))
    @test isturning(TurnStatistic(p₂, p₂, ρ₁))
    @test isturning(TurnStatistic(p₂, p₂, ρ₁))
end

@testset "low-level divergence statistics" begin
    a(p, divergent = false) = DivergenceStatistic(divergent, p, 1)
    x = a(0.3)
    @test get_acceptance_rate(x) ≈ 0.3
    y = a(0.6)
    @test get_acceptance_rate(y) ≈ 0.6
    z = reduce(combine_divstats, [x, x, y])
    @test get_acceptance_rate(z) ≈ 0.4
end


# test proposals

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
            @test prop.ω ≈ logaddexp(prop1.ω, prop2.ω)
        end
        @test count / N ≈ prob_prob2 atol = atol
    end
    prop1 = Proposal(1, log(1.0))
    prop2 = Proposal(2, log(3.0))
    prop3 = Proposal(3, log(1/3))
    test_sample(RNG, prop1, prop2, true, 1; atol = 0, N = 100)
    test_sample(RNG, prop1, prop2, false, 0.75)
    test_sample(RNG, prop1, prop3, true, 1/3)
    test_sample(RNG, prop1, prop3, false, 0.25)
end
