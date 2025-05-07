"""
$(SIGNATURES)

Random positive definite matrix of size `n` x `n` (for testing).
"""
function rand_Σ(::Type{Symmetric}, n)
    A = randn(RNG, n, n)
    Symmetric(A'*A .+ 0.01)
end

rand_Σ(::Type{Diagonal}, n) = Diagonal(randn(n).^2 .+ 0.01)

rand_Σ(n::Int) = rand_Σ(Symmetric, n)

"""
$(SIGNATURES)

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
    μ = [2, 1.2]
    D = [2.0, 0.7]
    m, C = simulated_meancov(()-> randn(2) .* D .+ μ, 10000)
    @test m ≈ μ atol = 0.05 rtol = 0.1
    @test C ≈ Diagonal(abs2.(D)) atol = 0.05 rtol = 0.1
end

###
### Hamiltonian test helper functions
###

"""
$(SIGNATURES)

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
find_stable_ϵ(κ::GaussianKineticEnergy, Σ) = √eigmin(κ.W'*Σ*κ.W)

"Multivariate normal with `Σ = LL'`."
multivariate_normal(μ, L) = shift(μ, linear(L, StandardMultivariateNormal(length(μ))))

"Multivariate normal with diagonal `Σ` (constant `v` variance)."
multivariate_normal(μ, v::Real = 1) = multivariate_normal(μ, I * v)

"""
$(SIGNATURES)

A `NamedTuple` that contains

- a random `K`-element vector `μ`

- a random `K × K` covariance matrix `Σ`,

- a random Hamiltonian `H` with `ℓ` corresponding to a multivariate normal with `μ`, `Σ`,
  and a random Gaussian kinetic energy (unrelated to `ℓ`).

- a random phasepoint `z`.

Useful for testing.
"""
function rand_Hz(K)
    μ = randn(K)
    Σ = rand_Σ(K)
    L = cholesky(Σ).L
    κ = GaussianKineticEnergy(inv(rand_Σ(Diagonal, K)))
    ℓ = multivariate_normal(μ, L)
    H = Hamiltonian(κ, ℓ)
    q = rand(RNG, ℓ)
    p = rand_p(RNG, κ)
    z = PhasePoint(evaluate_ℓ(H.ℓ, q), rand_p(RNG, κ))
    (μ = μ, Σ = Σ, H = H, z = z)
end
