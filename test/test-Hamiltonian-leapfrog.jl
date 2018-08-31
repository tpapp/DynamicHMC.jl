
# utility functions

"Test `loggradient` vs autodiff `neg_energy`."
function test_loggradient(ℓ, x)
    ∇ = DynamicHMC.loggradient(ℓ, x)
    ∇_AD = ForwardDiff.gradient(x->neg_energy(ℓ,x), x)
    @test ∇ ≈ ∇_AD
end

"""
    $SIGNATURES

Return a reasonable estimate for the largest stable stepsize (which may not be
stable, but is a good starting point for finding that).

`q` is assumed to be normal with variance `Σ`. `κ` is the kinetic energy.

Using the transformation ``p̃ = W⁻¹ p``, the kinetic energy is

``p'M⁻¹p = p'W⁻¹'W⁻¹p/2=p̃'p̃/2``

Transforming to ``q̃=W'q``, the variance of which becomes ``W' Σ W``. Return the
square root of its smallest eigenvalue, following Neal (2011, p 136).

When ``Σ⁻¹=M=WW'``, this the variance of `q̃` is ``W' Σ W=W' W'⁻¹W⁻¹W=I``, and
thus decorrelates the density perfectly.
"""
find_stable_ϵ(κ::GaussianKE, Σ) = √eigmin(κ.W'*Σ*κ.W)

find_stable_ϵ(H::Hamiltonian{<: DistributionLogDensity}) =
    find_stable_ϵ(H.κ, cov(H.ℓ.distribution))

"""
    $SIGNATURES

Simulated mean and covariance of `N` values from `f()`.
"""
function simulated_meancov(f, N)
    s = f()
    K = length(s)
    x = similar(s, (N, K))
    for i in 1:N
        x[i, :] = f()
    end
    m, C = mean_and_cov(x, 1)
    vec(m), C
end

@testset "simulated meancov" begin
    d = MvNormal([2,2], [2.0,1.0])
    m, C = simulated_meancov(()->rand(d), 10000)
    @test m ≈ mean(d) atol = 0.05 rtol = 0.1
    @test C ≈ cov(d) atol = 0.05 rtol = 0.1
end


# testsets

@testset "Gaussian KE full" begin
    for _ in 1:100
        K = rand(2:10)
        Σ = rand_Σ(Symmetric, K)
        κ = GaussianKE(inv(Σ))
        @test κ.Minv * κ.W * κ.W' ≈ Diagonal(ones(K))
        m, C = simulated_meancov(()->rand(κ), 10000)
        @test Matrix(Σ) ≈ C rtol = 0.1
        test_loggradient(κ, randn(K))
    end
end

@testset "Gaussian KE diagonal" begin
    for _ in 1:100
        K = rand(2:10)
        Σ = rand_Σ(Diagonal, K)
        κ = GaussianKE(inv(Σ))
        # FIXME workaround for https://github.com/JuliaLang/julia/issues/28869
        @test κ.Minv * Matrix(κ.W) * Matrix(κ.W') ≈ Diagonal(ones(K))
        m, C = simulated_meancov(()->rand(κ), 10000)
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
        ℓ2 = logdensity(ValueGradient, ℓ, q)
        @test ℓ2.value == ℓq.value
        @test ℓ2.gradient == ℓq.gradient
    end
    H, z = rand_Hz(rand(3:10))
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
        u = .√(1 ./ m)
        pₕ = p + ϵ/2*logdensity(ValueGradient, ℓ, q).gradient
        q′ = q + ϵ * u .* (u .* pₕ) # mimic numerical calculation leapfrog performs
        p′ = pₕ + ϵ/2*logdensity(ValueGradient, ℓ, q′).gradient
        q′, p′
    end

    n = 3
    M = rand_Σ(Diagonal, n)
    m = diag(M)
    κ = GaussianKE(inv(M))
    q = randn(n)
    p = randn(n)
    Σ = rand_Σ(n)
    ℓ = DistributionLogDensity(MvNormal(randn(n), Matrix(Σ)))
    H = Hamiltonian(ℓ, κ)
    ϵ = find_stable_ϵ(H)
    ℓq = logdensity(ValueGradient, ℓ, q)
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

@testset "PhasePoint validation and infinite values" begin
    @test_throws ArgumentError PhasePoint([1.0], [1.0, 2.0], # wrong p length
                                          ValueGradient(1.0, [1.0, 2.0]))
    @test_throws ArgumentError PhasePoint([1.0, 2.0], [1.0, 2.0], # wrong gradient length
                                          ValueGradient(1.0, [1.0]))
    @test PhasePoint([1.0], [1.0], ValueGradient(-Inf, [1.0])) isa PhasePoint
    @test neg_energy(Hamiltonian(DistributionLogDensity(MvNormal, 1), GaussianKE(1)),
                     PhasePoint([1.0], [1.0], ValueGradient(-Inf, [1.0]))) == -Inf
end

@testset "Hamiltonian and KE printing" begin
    κ = GaussianKE(Diagonal([1.0, 4.0]))
    @test repr(κ) == "Gaussian kinetic energy, √diag(M⁻¹): [1.0, 2.0]"
    H = Hamiltonian(DistributionLogDensity(MvNormal, 1), κ)
    @test repr(H) ==
        "Hamiltonian with Gaussian kinetic energy, √diag(M⁻¹): [1.0, 2.0]"
end
