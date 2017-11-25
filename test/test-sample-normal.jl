import DynamicHMC:
    NUTS_transition, NUTS_init, StepsizeTuner, StepsizeCovTuner, tune

@testset "unit normal simple HMC" begin
    # just testing leapfrog
    K = 2
    ℓ = MvNormal(zeros(K), ones(K))
    q = rand(ℓ)
    H = Hamiltonian(ℓ, GaussianKE(Diagonal(ones(K))))
    qs = sample_HMC(RNG, H, q, 10000)
    m, C = mean_and_cov(qs, 1)
    @test vec(m) ≈ zeros(K) atol = 0.1
    @test C ≈ full(Diagonal(ones(K))) atol = 0.1
end

@testset "normal NUTS HMC transition mean and cov" begin
    # a perfectly adapted Gaussian KE, should provide excellent mixing
    for _ in 1:100
        K = rand(2:8)
        N = 10000
        Σ = rand_Σ(K)
        ℓ = MvNormal(randn(K), full(Σ))
        q = rand(ℓ)
        H = Hamiltonian(ℓ, GaussianKE(Σ))
        qs = Array{Float64}(N, K)
        ϵ = 0.5
        for i in 1:N
            trans = NUTS_transition(RNG, H, q, ϵ, 5)
            q = trans.q
            qs[i, :] = q
        end
        m, C = mean_and_cov(qs, 1)
        @test vec(m) ≈ mean(ℓ) atol = 0.1 rtol = maximum(diag(C))*0.02 norm = x->vecnorm(x,1)
        @test cov(qs, 1) ≈ cov(ℓ) atol = 0.1 rtol = 0.1
    end
end

@testset "tuning building blocks" begin
    K = 4
    ℓ = MvNormal(zeros(K), fill(2.0, K))
    q = randn(K)
    sampler = NUTS_init(RNG, ℓ)
    tuner = StepsizeTuner(100)
    sampler2 = tune(sampler, tuner)
    tuner2 = StepsizeCovTuner(200, 10)
    sampler3 = tune(sampler, tuner2)
    @test all(diag(sampler3.H.κ.Minv) .≥ 2)
end

@testset "transition accessors and consistency checks" begin
    K = 2
    Σ = rand_Σ(K)
    ℓ = MvNormal(randn(K), full(Σ))
    q = rand(ℓ)
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
