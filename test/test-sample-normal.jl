@testset "unit normal simple HMC" begin
    function simple_HMC(rng, H, z::PhasePoint, ϵ, L)
        π₀ = logdensity(H, z)
        z′ = z
        for _ in 1:L
            z′ = leapfrog(H, z′, ϵ)
        end
        Δ = logdensity(H, z′) - π₀
        accept = Δ > 0 || (rand(rng) < exp(Δ))
        accept ? z′ : z
    end
    # this is just testing leapfrog
    K = 2
    N = 10000
    ℓ = MvNormal(zeros(K), ones(K))
    q = rand(ℓ)
    H = Hamiltonian(ℓ, GaussianKE(Diagonal(ones(K))))
    qs = Array{Float64}(N, K)
    ϵ = 0.2
    for i in 1:N
        z = rand_phasepoint(RNG, H, q)
        q = simple_HMC(RNG, H, z, rand_bool(RNG, 0.5) ? ϵ : -ϵ, 10).q
        qs[i, :] = q
    end
    m, C = mean_and_cov(qs, 1)
    @test vec(m) ≈ zeros(K) atol = 0.1
    @test C ≈ full(Diagonal(ones(K))) atol = 0.1
end

@testset "normal NUTS HMC transition mean and cov" begin
    # this is a perfectly adapted Gaussian KE, should provide excellent mixing
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
            trans = HMC_transition(RNG, H, q, ϵ, 5)
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
    sampler = TunedNUTS_init(RNG, ℓ, q)
    tuner = TunerStepsize(100)
    sampler2 = tune(RNG, sampler, tuner)
    tuner2 = TunerStepsizeCov(200, 10)
    sampler3 = tune(RNG, sampler, tuner2)
    @test all(diag(sampler3.H.κ.Minv) .≥ 2)
end
