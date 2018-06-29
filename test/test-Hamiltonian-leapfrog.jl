using DynamicHMC:
    GaussianKE, Hamiltonian, PhasePoint, neg_energy, phasepoint_in,
    rand_phasepoint, leapfrog, move, is_valid_ℓq, InitialStepsizeSearch,
    find_initial_stepsize

using DiffResults: MutableDiffResult

######################################################################
# Hamiltonian and leapfrog
######################################################################

@testset "Gaussian KE full" begin
    for _ in 1:100
        K = rand(RNG, 2:10)
        Σ = rand_Σ(RNG, Symmetric, K)
        κ = GaussianKE(inv(Σ))
        @test κ.Minv * κ.W * κ.W' ≈ Diagonal(ones(K))
        m, C = simulated_meancov(()->rand(RNG, κ), 10000)
        @test Matrix(Σ) ≈ C rtol = 0.1
        test_loggradient(κ, randn(K))
    end
end

@testset "Gaussian KE diagonal" begin
    for _ in 1:100
        K = rand(2:10)
        Σ = rand_Σ(RNG, Diagonal, K)
        κ = GaussianKE(inv(Σ))
        @test κ.Minv * κ.W * κ.W' ≈ Diagonal(ones(K))
        m, C = simulated_meancov(()->rand(RNG, κ), 10000)
        @test Matrix(Σ) ≈ C rtol = 0.1
        test_loggradient(κ, randn(K))
    end
end

@testset "phasepoint internal consistency" begin
    # when this breaks, interface was modified, rewrite tests
    @test fieldnames(PhasePoint) == (:q, :p, :ℓq)
    "Test the consistency of cached values."
    function test_consistency(H, z)
        @unpack q, ℓq = z
        @unpack ℓ = H
        @test ℓ(q) == ℓq
    end
    H, z = rand_Hz(RNG, rand(3:10))
    test_consistency(H, z)
    ϵ = find_stable_ϵ(H)
    for _ in 1:10
        z = leapfrog(H, z, ϵ)
        test_consistency(H, z)
    end
end

@testset "leapfrog" begin
    """
    Simple leapfrog implementation. `q`: position, `p`: momentum, `ℓ`: neg_energy, `ϵ`: stepsize. `m` is the diagonal of the kinetic energy ``K(p)=p'M⁻¹p``, defaults to `1`.
    """
    function leapfrog_Gaussian(q, p, ℓ, ϵ, m = ones(length(p)))
        u = @. √(1 / m)
        pₕ = p + ϵ/2*DiffResults.gradient(ℓ(q))
        q′ = q + ϵ * u .* (u .* pₕ) # mimic numerical calculation leapfrog performs
        p′ = pₕ + ϵ/2*DiffResults.gradient(ℓ(q′))
        q′, p′
    end

    n = 3
    M = rand_Σ(RNG, Diagonal, n)
    m = diag(M)
    κ = GaussianKE(inv(M))
    q = randn(RNG, n)
    p = randn(RNG, n)
    Σ = rand_Σ(RNG, n)
    ℓ = MultiNormal(randn(RNG, n), Σ)
    H = Hamiltonian(ℓ, κ)
    ϵ = find_stable_ϵ(H)
    ℓq = ℓ(q)
    q₂, p₂ = copy(q), copy(p)
    q′, p′ = leapfrog_Gaussian(q, p, ℓ, ϵ, m)
    z = PhasePoint(q, p, ℓq)
    z′ = leapfrog(H, z, ϵ)

    ⩳(x, y) = isapprox(x, y, rtol = √eps(), atol = √eps())

    @test p == p₂               # arguments not modified
    @test q == q₂
    @test z′.q ⩳ q′
    @test z′.p ⩳ p′

    for i in 1:100
        q, p = leapfrog_Gaussian(q, p, ℓ, ϵ, m)
        z = leapfrog(H, z, ϵ)
        @test z.q ⩳ q
        @test z.p ⩳ p
    end
end

@testset "leapfrog" begin
    "Test that the Hamiltonian is invariant using the leapfrog integrator."
    function test_hamiltonian_invariance(H, z, L, ϵ; atol = one(ϵ))
        π₀ = neg_energy(H, z)
        warned = false
        for i in 1:L
            z = leapfrog(H, z, ϵ)
            Δ = π₀ - neg_energy(H, z)
            if abs(Δ) ≥ atol && !warned
                warn("Hamiltonian invariance violated: step $(i) of $(L), Δ = $(Δ)")
                show(H)
                show(z)
                warned = true
            end
            @test Δ ≈ 0 atol = atol
        end
    end

    for _ in 1:100
        H, z = rand_Hz(RNG, rand(2:5))
        ϵ = find_initial_stepsize(InitialStepsizeSearch(), H, z)
        test_hamiltonian_invariance(H, z, 100, ϵ/20; atol = 2.0)
    end
end

@testset "PhasePoint validation and infinite values" begin
    @test PhasePoint([1.0], [1.0], MutableDiffResult(1.0, (1.0,))) isa PhasePoint
    @test_throws ArgumentError PhasePoint([1.0], [1.0, 2.0],
                                          MutableDiffResult(1.0, (1.0,)))
    @test PhasePoint([1.0], [1.0], MutableDiffResult(-Inf, (1.0,))) isa PhasePoint
    @test_throws DomainError PhasePoint([1.0], [1.0],
                                        MutableDiffResult(1.0, (-Inf,)))
    @test_throws DomainError PhasePoint([1.0], [1.0],
                                        MutableDiffResult(NaN, (1.0,)))
    @test_throws DomainError PhasePoint([1.0], [1.0],
                                        MutableDiffResult(Inf, (1.0,)))
    @test neg_energy(Hamiltonian(identity, GaussianKE(1)),
                     PhasePoint([1.0], [1.0], MutableDiffResult(-Inf, (1.0,)))) == -Inf
end

@testset "Hamiltonian and KE printing" begin
    κ = GaussianKE(Diagonal([1.0, 4.0]))
    @test repr(κ) == "Gaussian kinetic energy, √diag(M⁻¹): [1.0, 2.0]"
    H = Hamiltonian(identity, κ)
    @test repr(H) ==
        "Hamiltonian with Gaussian kinetic energy, √diag(M⁻¹): [1.0, 2.0]"
end
