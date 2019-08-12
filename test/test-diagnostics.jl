isinteractive() && include("common.jl")

#####
##### test diagnostics
#####

@testset "summarize tree statistics" begin
    N = 1000
    directions = Directions(UInt32(0))
    function rand_invalidtree()
        if rand() < 0.1
            REACHED_MAX_DEPTH
        else
            left = rand(-5:5)
            right = left + rand(0:5)
            InvalidTree(left, right)
        end
    end
    tree_statistics = [TreeStatisticsNUTS(randn(), rand(0:5), rand_invalidtree(), rand(),
                                          rand(1:30), directions) for _ in 1:N]
    stats = summarize_tree_statistics(tree_statistics)
    # acceptance rates
    @test stats.N == N
    @test stats.a_mean ≈ mean(x -> x.acceptance_rate, tree_statistics)
    @test stats.a_quantiles ==
        quantile((x -> x.acceptance_rate).(tree_statistics), ACCEPTANCE_QUANTILES)
    # termination counts
    @test stats.termination_counts.divergence ==
        count(x -> is_divergent(x.termination), tree_statistics)
    @test stats.termination_counts.max_depth ==
        count(x -> x.termination == REACHED_MAX_DEPTH, tree_statistics)
    @test stats.termination_counts.turning ==
        (N - stats.termination_counts.max_depth - stats.termination_counts.divergence)
    # depth counts
    for (i, c) in enumerate(stats.depth_counts)
        @test count(x -> x.depth == i - 1, tree_statistics) == c
    end
    @test sum(stats.depth_counts) == N
    # misc
    @test 1.8 ≤ EBFMI(tree_statistics) ≤ 2.2 # nonsensical value, just checking calculation
    @test repr(stats) isa AbstractString # just test that it prints w/o error
end

@testset "log acceptance ratios" begin
    ℓ = DistributionLogDensity(MvNormal(ones(5), Diagonal(ones(5))))
    log2ϵs = -5:5
    N = 13
    logA = explore_log_acceptance_ratios(ℓ, zeros(5), log2ϵs; N = N)
    @test all(isfinite.(logA))
    @test size(logA) == (length(log2ϵs), N)
end

@testset "leapfrog trajectory" begin
    # problem setup
    K = 2
    ℓ = DistributionLogDensity(MvNormal(ones(K), Diagonal(ones(K))))
    κ = GaussianKineticEnergy(K)
    q = zeros(K)
    Q = evaluate_ℓ(ℓ, q)
    p = ones(K) .* 0.98
    H = Hamiltonian(κ, ℓ)
    ϵ = 0.1
    ixs = 1:15
    ix0 = 5

    # calculate trajectory manually
    zs1 = let z = PhasePoint(Q, p)
        [(z1 = z; z = leapfrog(H, z, ϵ); z1) for _ in ixs]
    end
    πs1 = logdensity.(Ref(H), zs1)
    Δs1 = πs1 .- πs1[ix0]

    # calculate using function
    @unpack Δs, zs = leapfrog_trajectory(ℓ, zs1[ix0].Q.q, ϵ, ixs .- ix0; κ = κ,
                                         p = zs1[ix0].p)
    @test all(isapprox.(Δs, Δs1; atol = 1e-5))
    @test all(map((x, y) -> x.Q.q ≈ y.Q.q, zs, zs1))
    @test all(map((x, y) -> x.p ≈ y.p, zs, zs1))
end
