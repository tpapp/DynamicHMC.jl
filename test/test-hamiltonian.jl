####
#### utility functions
####

"Test kinetic energy gradient by automatic differentiation."
function test_KE_gradient(κ::DynamicHMC.EuclideanKineticEnergy, p)
    ∇ = ∇kinetic_energy(κ, p)
    ∇_AD = ForwardDiff.gradient(p -> kinetic_energy(κ, p), p)
    @test ∇ ≈ ∇_AD
end

####
#### testsets
####

@testset "Gaussian KE full" begin
    for _ in 1:100
        K = rand(2:10)
        Σ = rand_Σ(Symmetric, K)
        κ = GaussianKineticEnergy(inv(Σ))
        @unpack M⁻¹, W = κ
        @test W isa LowerTriangular
        @test M⁻¹ * W * W' ≈ Diagonal(ones(K))
        m, C = simulated_meancov(()->rand_p(RNG, κ), 10000)
        @test Matrix(Σ) ≈ C rtol = 0.1
        test_KE_gradient(κ, randn(K))
    end
end

@testset "Gaussian KE diagonal" begin
    for _ in 1:100
        K = rand(2:10)
        Σ = rand_Σ(Diagonal, K)
        κ = GaussianKineticEnergy(inv(Σ))
        @unpack M⁻¹, W = κ
        @test W isa Diagonal
        # FIXME workaround for https://github.com/JuliaLang/julia/issues/28869
        @test M⁻¹ * (W * W') ≈ Diagonal(ones(K))
        m, C = simulated_meancov(()->rand_p(RNG, κ), 10000)
        @test Matrix(Σ) ≈ C rtol = 0.1
        test_KE_gradient(κ, randn(K))
    end
end

@testset "phasepoint internal consistency" begin
    # when this breaks, interface was modified, rewrite tests
    @test fieldnames(PhasePoint) == (:Q, :p)
    "Test the consistency of cached values."
    function test_consistency(H, z)
        @unpack q, ℓq, ∇ℓq = z.Q
        @unpack ℓ = H
        ℓ2, ∇ℓ2 = logdensity_and_gradient(ℓ, q)
        @test ℓ2 == ℓq
        @test ∇ℓ2 == ∇ℓq
    end
    H, z = rand_Hz(rand(3:10))
    test_consistency(H, z)
    ϵ = find_stable_ϵ(H)
    for _ in 1:10
        z = leapfrog(H, z, ϵ)
        test_consistency(H, z)
    end
end

@testset "leapfrog calculation" begin
    # Simple leapfrog implementation. `q`: position, `p`: momentum, `ℓ`: neg_energy, `ϵ`:
    # stepsize. `m` is the diagonal of the kinetic energy ``K(p)=p'M⁻¹p``, defaults to `I`.
    function leapfrog_Gaussian(q, p, ℓ, ϵ, m = ones(length(p)))
        u = .√(1 ./ m)
        pₕ = p .+ ϵ/2 .* last(logdensity_and_gradient(ℓ, q))
        q′ = q .+ ϵ * u .* (u .* pₕ) # mimic numerical calculation leapfrog performs
        p′ = pₕ .+ ϵ/2 .* last(logdensity_and_gradient(ℓ, q′))
        q′, p′
    end

    n = 3
    M = rand_Σ(Diagonal, n)
    m = diag(M)
    κ = GaussianKineticEnergy(inv(M))
    q = randn(n)
    p = randn(n)
    Σ = rand_Σ(n)
    ℓ = DistributionLogDensity(MvNormal(randn(n), Matrix(Σ)))
    H = Hamiltonian(κ, ℓ)
    ϵ = find_stable_ϵ(H)
    z = PhasePoint(evaluate_ℓ(ℓ, q), p)

    @testset "arguments not modified" begin
        q₂, p₂ = copy(q), copy(p)
        q′, p′ = leapfrog_Gaussian(q, p, ℓ, ϵ, m)
        z′ = leapfrog(H, z, ϵ)
        @test p == p₂               # arguments not modified
        @test q == q₂
        @test z′.Q.q ≈ q′
        @test z′.p ≈ p′
    end

    @testset "leapfrog steps" begin
        for i in 1:100
            q, p = leapfrog_Gaussian(q, p, ℓ, ϵ, m)
            z = leapfrog(H, z, ϵ)
            @test z.Q.q ≈ q
            @test z.p ≈ p
        end
    end
end

@testset "leapfrog Hamiltonian invariance" begin
    "Test that the Hamiltonian is invariant using the leapfrog integrator."
    function test_hamiltonian_invariance(H, z, L, ϵ; atol = one(ϵ))
        π₀ = logdensity(H, z)
        warned = false
        for i in 1:L
            z = leapfrog(H, z, ϵ)
            Δ = logdensity(H, z) - π₀
            if abs(Δ) ≥ atol && !warned
                @warn "Hamiltonian invariance violated: step $(i) of $(L), Δ = $(Δ)"
                show(H)
                show(z)
                warned = true
            end
            @test Δ ≈ 0 atol = atol
        end
    end

    for _ in 1:100
        H, z = rand_Hz(rand(2:5))
        ϵ = find_initial_stepsize(InitialStepsizeSearch(), H, z)
        test_hamiltonian_invariance(H, z, 100, ϵ/20; atol = 2.0)
    end
end

@testset "leapfrog back and forth" begin
    for _ in 1:1000
        H, z = rand_Hz(5)
        z1 = z
        N = 5
        ϵ = 0.1
        z1 = leapfrog(H, z1, ϵ)
        z1 = leapfrog(H, z1, -ϵ)
        @test z.p ≈ z1.p norm = x -> norm(x, Inf)
        @test z.Q.q ≈ z1.Q.q norm = x -> norm(x, Inf)
    end

    for _ in 1:100
        H, z = rand_Hz(2)
        z1 = z
        N = 3
        ϵ = 0.1

        # forward
        for _ in 1:N
            z1 = leapfrog(H, z1, ϵ)
        end

        # backward
        for _ in 1:N
            z1 = leapfrog(H, z1, -ϵ)
        end

        @test z.p ≈ z1.p norm = x -> norm(x, Inf) rtol = 0.001
        @test z.Q.q ≈ z1.Q.q norm = x -> norm(x, Inf) rtol = 0.001
    end
end

@testset "PhasePoint validation and infinite values" begin
    # wrong gradient length
    @test_throws ArgumentError EvaluatedLogDensity([1.0, 2.0], 1.0, [1.0])
    # wrong p length
    Q = EvaluatedLogDensity([1.0, 2.0], 1.0, [1.0, 2.0])
    @test_throws ArgumentError PhasePoint(Q, [1.0])
    @test PhasePoint(Q, [1.0, 2.0]) isa PhasePoint
    # infinity fallbacks
    h = Hamiltonian(GaussianKineticEnergy(1), DistributionLogDensity(MvNormal, 1))
    @test logdensity(h, PhasePoint(EvaluatedLogDensity([1.0], -Inf, [1.0]), [1.0])) == -Inf
    @test logdensity(h, PhasePoint(EvaluatedLogDensity([1.0], NaN, [1.0]), [1.0])) == -Inf
    @test logdensity(h, PhasePoint(EvaluatedLogDensity([1.0], 9.0, [1.0]), [NaN])) == -Inf
end

@testset "Hamiltonian and KE printing" begin
    κ = GaussianKineticEnergy(Diagonal([1.0, 4.0]))
    @test repr(κ) == "Gaussian kinetic energy, √diag(M⁻¹): [1.0, 2.0]"
    H = Hamiltonian(κ, DistributionLogDensity(MvNormal, 1))
    @test repr(H) ==
        "Hamiltonian with Gaussian kinetic energy, √diag(M⁻¹): [1.0, 2.0]"
end
