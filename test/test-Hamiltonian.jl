"""
Test that the Hamiltonian is invariant using the leapfrog integrator.
"""
function test_hamiltonian_invariance(H, ϵ, L; atol=1e-3)
    q = rand(H.ℓ)
    p = propose(H.κ, q)
    z = PhasePoint(q, p)
    π₀ = logdensity(H, z)
    for _ in 1:L
        z = leapfrog(H, z, ϵ)
        @test isapprox(π₀, logdensity(H, z); atol = atol)
    end
end

@testset "leapfrog" begin
    test_hamiltonian_invariance(Hamiltonian(normal_density(fill(0.0, 3), I), 
                                            GaussianKE(Diagonal(ones(3)))),
                                0.01, 100)
    A = rand(3,3)
    test_hamiltonian_invariance(Hamiltonian(normal_density(randn(3), A'*A), 
                                            GaussianKE(Diagonal([1.0,0.5,2.0]))),
                                            0.001, 100)
end
