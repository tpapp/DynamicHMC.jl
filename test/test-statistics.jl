import DynamicHMC:
    NUTS_Transition, Termination, ACCEPTANCE_QUANTILES, NUTS_statistics,
    get_acceptance_rate, get_termination, get_depth

@testset "NUTS statistics" begin
    N = 1000
    t = collect(instances(Termination))
    sample = [NUTS_Transition([1.0], randn(), rand(1:5), rand(t), rand(), rand(1:30))
               for _ in 1:N]
    stats = NUTS_statistics(sample)
    @test stats.N == N
    @test stats.a_mean ≈ mean(get_acceptance_rate, sample)
    @test stats.a_quantiles == quantile(get_acceptance_rate.(sample), ACCEPTANCE_QUANTILES)
    @test stats.termination_counts == counter(map(get_termination, sample))
    @test stats.depth_counts == counter(map(get_depth, sample))
    @test 1.8 ≤ EBFMI(sample) ≤ 2.2 # nonsensical value, just checking calculation
    @test repr(stats) isa AbstractString # just test that it prints w/o error
end
