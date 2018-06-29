import DynamicHMC:
    NUTS_transition, NUTS_init, StepsizeTuner, StepsizeCovTuner, tune

@testset "unit normal simple HMC" begin
    # just testing leapfrog
    K = 2
    μ = zeros(K)
    Σ = Diagonal(ones(K))
    ℓ = MultiNormal(μ, Σ)
    q = rand(RNG, ℓ)
    H = Hamiltonian(ℓ, GaussianKE(Diagonal(ones(K))))
    qs = sample_HMC(RNG, H, q, 10000)
    m, C = mean_and_cov(qs, 1)
    @test reldiff(μ, m) ≤ 0.1
    @test reldiff(Σ, C) ≤ 0.1
end

@testset "normal NUTS HMC transition mean and cov" begin
    # a perfectly adapted Gaussian KE, should provide excellent mixing
    for _ in 1:100
        K = rand(RNG, 2:8)
        N = 10000
        μ = randn(K)
        Σ = rand_Σ(RNG, K)
        ℓ = MultiNormal(μ, Σ)
        q = rand(RNG, ℓ)
        H = Hamiltonian(ℓ, GaussianKE(Σ))
        qs = Array{Float64}(undef, N, K)
        ϵ = 0.5
        for i in 1:N
            trans = NUTS_transition(RNG, H, q, ϵ, 5)
            q = trans.q
            qs[i, :] = q
        end
        m, C = mean_and_cov(qs, 1)
        # @test μ ≈ vec(m) atol = 0.1 rtol = maximum(diag(C))*0.02 norm = x->norm(x,1)
        @test reldiff(μ, vec(m)) ≤ 0.1
        @test reldiff(Σ, C, 2) ≤ 0.2
    end
end

@testset "tuning building blocks" begin
    K = 4
    ℓ = MultiNormal(zeros(K), Diagonal(fill(2.0, K)))
    sampler = NUTS_init(RNG, ℓ, K)
    tuner = StepsizeTuner(100)
    sampler2 = tune(sampler, tuner)
    tuner2 = StepsizeCovTuner(200, 10)
    sampler3 = tune(sampler, tuner2)
    @test_skip all(diag(sampler3.H.κ.Minv) .≥ 2)
end

@testset "transition accessors and consistency checks" begin
    K = 2
    Σ = rand_Σ(RNG, K)
    ℓ = MultiNormal(randn(RNG, K), Σ)
    q = rand(RNG, ℓ)
    H = Hamiltonian(ℓ, GaussianKE(Σ))
    ϵ = 0.5
    valid_terminations = [DynamicHMC.MaxDepth,
                          DynamicHMC.AdjacentDivergent,
                          DynamicHMC.AdjacentTurn,
                          DynamicHMC.DoubledTurn]
    for _ in 1:1000
        trans = NUTS_transition(RNG, H, q, ϵ, 5)
        q′ = get_position(trans)
        @test q′ isa Vector{Float64}
        @test length(q′) == length(q)
        @test get_neg_energy(trans) isa Float64
        depth = get_depth(trans)
        @test depth isa Int
        @test depth ≥ 1
        @test get_termination(trans) ∈ valid_terminations
        a = get_acceptance_rate(trans)
        @test a isa Float64
        @test 0 ≤ a ≤ 1
        steps = get_steps(trans)
        @test steps isa Int
        @test steps ≥ 1
    end
end
