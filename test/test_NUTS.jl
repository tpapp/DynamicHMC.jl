using DynamicHMC: TrajectoryNUTS, rand_bool_logprob, GeneralizedTurnStatistic,
    AcceptanceStatistic, leaf_acceptance_statistic, acceptance_rate, TreeStatisticsNUTS,
    NUTS, sample_tree, combine_turn_statistics, combine_visited_statistics

####
#### utilities
####

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


###
### random booleans
###

@testset "random booleans" begin
    for prob in (1:9) ./ 10
        logprob = log(prob)
        @test abs(mean(rand_bool_logprob(RNG, logprob) for _ in 1:10000) - prob) ≤ 0.02
    end

    # these operations don't call the RNG, this is checked
    RNG′ = copy(RNG)
    @test all(rand_bool_logprob(RNG, 0) for _ in 1:10000)
    @test all(rand_bool_logprob(RNG, 10) for _ in 1:10000)
    @test rand(RNG′) == rand(RNG)
end

###
### test turn statistics
###

@testset "low-level turn statistics" begin
    trajectory = TrajectoryNUTS(nothing, 0, 1, -1000, Val{:generalized})
    p = ones(3)                 # unit vector
    c = 0.1                     # a constant, just for consistency checking of combination
    # turn statistics constructed so that τ₁ + τ₂ won't be turning, τ₁ + τ₃ will be
    τ₁ = GeneralizedTurnStatistic(p, p .- c, p, p .- c, p)
    τ₂ = GeneralizedTurnStatistic(3 .* p, 3 .* p .+ c, 3 .* p, 3 .* p .+ c, 3 .* p)
    τ₃ = GeneralizedTurnStatistic(2 .* p, 2 .* p .+ c, 2 .* p, 2 .* p .+ c, -2 .* p)
    τ = combine_turn_statistics(trajectory, τ₁, τ₂)
    # test mechanics of combination
    @test τ ≂ GeneralizedTurnStatistic(τ₁.p₋, τ₁.p♯₋, τ₂.p₊, τ₂.p♯₊, τ₁.ρ .+ τ₂.ρ)
    # test non-turning
    @test !is_turning(trajectory, τ)
    # test turning
    @test is_turning(trajectory, combine_turn_statistics(trajectory, τ₁, τ₃))
end

@testset "low-level visited statistics" begin
    trajectory = TrajectoryNUTS(nothing, 0, 1, -1000, Val{:generalized})
    vs(p, is_initial = false) = leaf_acceptance_statistic(log(p), is_initial)
    x = vs(0.3)
    @test acceptance_rate(x) ≈ 0.3
    y = vs(0.6)
    @test acceptance_rate(y) ≈ 0.6
    x0 = vs(10, true)    # initial node, does not count
    z = reduce((x, y) -> combine_visited_statistics(trajectory, x, y),
               [x, x, y, x0])
    @test acceptance_rate(z) ≈ 0.4
end

# define a distribution which is divergent everywhere except at 0
struct AlwaysDivergentTest
    K::Int
end

function LogDensityProblems.capabilities(::Type{AlwaysDivergentTest})
    LogDensityProblems.LogDensityOrder{1}()
end
LogDensityProblems.dimension(d::AlwaysDivergentTest) = d.K
function LogDensityProblems.logdensity_and_gradient(d::AlwaysDivergentTest, x)
    ∇ = ones(length(x))
    if all(iszero.(x))
        0.0, ∇
    else
        -Inf, ∇
    end
end

@testset "unconditional divergence" begin
    # test NUTS sampler where all movements are divergent
    K = 3
    ℓ = AlwaysDivergentTest(K)
    Q, tree_statistics = sample_tree(RNG, NUTS(), Hamiltonian(GaussianKineticEnergy(K), ℓ),
                                     evaluate_ℓ(ℓ, zeros(K)), 1.0)
    @test is_divergent(tree_statistics.termination)
    @test iszero(tree_statistics.acceptance_rate)
    @test iszero(tree_statistics.depth)
    @test tree_statistics.steps == 1
end

@testset "normal NUTS HMC transition mean and cov" begin
    # A test for sample_tree with a fixed ϵ and κ, which is perfectly adapted and should
    # provide excellent mixing
    for _ in 1:10
        K = rand(2:8)
        N = 10000
        μ = randn(K)
        Σ = rand_Σ(K)
        L = cholesky(Σ).L
        ℓ = multivariate_normal(μ, L)
        Q = evaluate_ℓ(ℓ, randn(K))
        H = Hamiltonian(GaussianKineticEnergy(Σ), ℓ)
        qs = Array{Float64}(undef, N, K)
        ϵ = 0.5
        algorithm = NUTS()
        for i in 1:N
            Q = first(sample_tree(RNG, algorithm, H, Q, ϵ))
            qs[i, :] = Q.q
        end
        m, C = mean_and_cov(qs, 1)
        @test vec(m) ≈ μ atol = 0.1 rtol = maximum(diag(C))*0.02 norm = x -> norm(x,1)
        @test cov(qs, dims = 1) ≈ L*L' atol = 0.1 rtol = 0.1
    end
end

###
### test proposals
###

# NOTE superseded by separating ω and ζ
# @testset "proposal" begin
#     trajectory = TrajectoryNUTS(nothing, 0, 1, -1000)
#     function test_sample(rng, prop1, prop2, bias, prob_prob2; atol = 0.02, N = 10000)
#         count = 0
#         for _ in 1:N
#             prop = combine_proposals(rng, trajectory, prop1, prop2,
#                                      # direction irrelevant for thie method
#                                      rand(Bool),
#                                      # test with given bias
#                                      bias)
#             if prop.z ≂ prop2.z
#                 count += 1
#             else
#                 @test prop.z ≂ prop1.z
#             end
#             @test prop.ω ≈ logaddexp(prop1.ω, prop2.ω)
#         end
#         @test count / N ≈ prob_prob2 atol = atol
#     end
#     prop1 = Proposal(1, log(1.0))
#     prop2 = Proposal(2, log(3.0))
#     prop3 = Proposal(3, log(1/3))
#     test_sample(RNG, prop1, prop2, true, 1; atol = 0, N = 100)
#     test_sample(RNG, prop1, prop2, false, 0.75)
#     test_sample(RNG, prop1, prop3, true, 1/3)
#     test_sample(RNG, prop1, prop3, false, 0.25)
# end
